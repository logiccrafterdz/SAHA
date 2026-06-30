"""
SAHA – ClaudeAdapter integration test (mock HTTP).
Tests the *real* ClaudeAdapter code path end-to-end without a live API key.
We patch anthropic.AsyncAnthropic so no network call is made, but all
translation logic (request building, response parsing, cost calc, tool
detection) runs exactly as in production.

Also includes a full pipeline smoke test:
  ClaudeAdapter (mocked) → AgentLoop → NormalizationPipeline → Grader
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saha.contracts.common import CanonicalError, new_uuid
from saha.contracts.eval import EvalInput, SuccessContract
from saha.contracts.execution import AgentState, AgentStatus
from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderCapabilities,
    ProviderPolicies,
    ProviderProfile,
    UnifiedAgentRequest,
)
from saha.eval.grader import Grader
from saha.eval.normalizer import NormalizationPipeline
from saha.event_bus import topics
from saha.execution.agent_loop import AgentLoop, TaskSpec
from saha.vendor import VendorGateway
from saha.vendor.adapters.claude import ClaudeAdapter


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_anthropic_message(
    text: str = "Task completed successfully.",
    input_tokens: int = 120,
    output_tokens: int = 60,
    stop_reason: str = "end_turn",
    tool_use: dict | None = None,
) -> MagicMock:
    """Build a fake anthropic.types.Message matching the SDK response shape."""
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.usage = MagicMock()
    msg.usage.input_tokens  = input_tokens
    msg.usage.output_tokens = output_tokens

    if tool_use:
        tool_block       = MagicMock()
        tool_block.type  = "tool_use"
        tool_block.name  = tool_use["name"]
        tool_block.input = tool_use["arguments"]
        tool_block.id    = f"toolu_{new_uuid()[:8]}"
        msg.content = [tool_block]
    else:
        text_block       = MagicMock()
        text_block.type  = "text"
        text_block.text  = text
        msg.content = [text_block]

    return msg


def _patch_claude(response_msg: MagicMock) -> Any:
    """Context manager: patches AsyncAnthropic.messages.create with a fixed response."""
    mock_client      = MagicMock()
    mock_messages    = AsyncMock()
    mock_messages.create = AsyncMock(return_value=response_msg)
    mock_client.messages = mock_messages

    return patch(
        "saha.vendor.adapters.claude.anthropic.AsyncAnthropic",
        return_value=mock_client,
    )


# ─── State / Bus mocks (reused from test_end_to_end) ─────────────────────────

class MockBus:
    def __init__(self) -> None:
        self.published: dict[str, list[dict]] = defaultdict(list)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published[topic].append(payload)

    async def subscribe(self, topic: str, handler: Any) -> None:
        pass


class MockStateManager:
    def __init__(self) -> None:
        self._store: dict[str, AgentState] = {}

    async def create(self, task_id: str, provider_id: str, budget_cap: float = 5.0) -> AgentState:
        state = AgentState(task_id=task_id, provider_id=provider_id, budget_cap=budget_cap)
        self._store[state.agent_state_id] = state
        return state

    async def get(self, agent_state_id: str) -> AgentState | None:
        return self._store.get(agent_state_id)

    async def update(self, state: AgentState) -> None:
        self._store[state.agent_state_id] = state

    async def mark_completed(self, agent_state_id: str) -> None:
        if s := self._store.get(agent_state_id):
            s.status = AgentStatus.COMPLETED

    async def mark_failed(self, agent_state_id: str) -> None:
        if s := self._store.get(agent_state_id):
            s.status = AgentStatus.FAILED


def _make_gateway_from_adapter(adapter: ClaudeAdapter) -> VendorGateway:
    gw = VendorGateway.__new__(VendorGateway)
    gw._adapters = {adapter.provider_id: adapter}
    return gw


# ─── ClaudeAdapter unit tests (with mocked HTTP) ─────────────────────────────

class TestClaudeAdapterTextCompletion:
    """Test the adapter's translation logic: SAHA request → Anthropic → SAHA response."""

    async def test_simple_text_response(self) -> None:
        """COMPLETED response: text block → normalized_output.text, cost > 0."""
        fake_msg = _make_anthropic_message(
            text="def sort_list(lst): return sorted(lst)",
            input_tokens=200,
            output_tokens=30,
        )
        with _patch_claude(fake_msg):
            adapter = ClaudeAdapter()
            request = UnifiedAgentRequest(
                task_id=new_uuid(),
                agent_state_id=new_uuid(),
                message="Write a Python sort function",
                system_prompt="You are a Python expert.",
            )
            response = await adapter.complete(request)

        assert response.status == "COMPLETED"
        assert "sort_list" in response.normalized_output["text"]
        assert response.context_tokens_used == 230  # 200 + 30
        # Cost: (200 * 3.0 + 30 * 15.0) / 1_000_000 = 0.00105
        assert response.cost_estimate == pytest.approx(0.00105, rel=1e-3)
        assert response.error.type == "NONE"
        assert response.latency_ms >= 0

    async def test_cost_calculation_correctness(self) -> None:
        """Verify cost formula: (input*3 + output*15) / 1_000_000."""
        fake_msg = _make_anthropic_message(input_tokens=1_000_000, output_tokens=1_000_000)
        with _patch_claude(fake_msg):
            adapter  = ClaudeAdapter()
            request  = UnifiedAgentRequest(task_id="t", agent_state_id="a", message="x")
            response = await adapter.complete(request)
        assert response.cost_estimate == pytest.approx(18.0, rel=1e-3)  # 3+15

    async def test_provider_id_is_stable(self) -> None:
        """provider_id in response must match the adapter's stable ID."""
        fake_msg = _make_anthropic_message()
        with _patch_claude(fake_msg):
            adapter  = ClaudeAdapter()
            request  = UnifiedAgentRequest(task_id="t", agent_state_id="a", message="hi")
            response = await adapter.complete(request)
        assert response.provider_id == "claude_3_5_sonnet"

    async def test_request_id_echoed(self) -> None:
        """request_id from UnifiedAgentRequest must be echoed in the response."""
        fake_msg = _make_anthropic_message()
        with _patch_claude(fake_msg):
            adapter = ClaudeAdapter()
            req_id  = new_uuid()
            request = UnifiedAgentRequest(
                request_id=req_id, task_id="t", agent_state_id="a", message="hi"
            )
            response = await adapter.complete(request)
        assert response.request_id == req_id


class TestClaudeAdapterToolUse:
    """Test NEEDS_TOOL detection and pending_tool_call population."""

    async def test_tool_use_detected(self) -> None:
        """When the model returns a tool_use block, status must be NEEDS_TOOL."""
        fake_msg = _make_anthropic_message(
            tool_use={"name": "read_file", "arguments": {"path": "/src/main.py"}},
        )
        with _patch_claude(fake_msg):
            adapter  = ClaudeAdapter()
            request  = UnifiedAgentRequest(task_id="t", agent_state_id="a", message="fix bug")
            response = await adapter.complete(request)

        assert response.status == "NEEDS_TOOL"
        assert response.pending_tool_call is not None
        assert response.pending_tool_call["name"] == "read_file"
        assert response.pending_tool_call["arguments"]["path"] == "/src/main.py"
        assert "tool_use_id" in response.pending_tool_call
        assert response.tool_calls_count == 1

    async def test_tool_use_cost_still_computed(self) -> None:
        """Cost is estimated even on NEEDS_TOOL responses."""
        fake_msg = _make_anthropic_message(
            tool_use={"name": "echo", "arguments": {"message": "test"}},
            input_tokens=500,
            output_tokens=80,
        )
        with _patch_claude(fake_msg):
            adapter  = ClaudeAdapter()
            request  = UnifiedAgentRequest(task_id="t", agent_state_id="a", message="ping")
            response = await adapter.complete(request)

        expected = (500 * 3.0 + 80 * 15.0) / 1_000_000
        assert response.cost_estimate == pytest.approx(expected, rel=1e-3)


class TestClaudeAdapterErrorHandling:
    """Test that exceptions are caught and returned as CanonicalError (never raised)."""

    async def test_api_exception_returns_failed_response(self) -> None:
        """Any SDK exception → FAILED response with populated error field."""
        mock_client   = MagicMock()
        mock_messages = AsyncMock()
        mock_messages.create = AsyncMock(side_effect=RuntimeError("Network error"))
        mock_client.messages = mock_messages

        with patch("saha.vendor.adapters.claude.anthropic.AsyncAnthropic", return_value=mock_client):
            adapter  = ClaudeAdapter()
            request  = UnifiedAgentRequest(task_id="t", agent_state_id="a", message="x")
            response = await adapter.complete(request)

        assert response.status == "FAILED"
        assert response.error.type != "NONE"
        assert response.error.details != ""

    async def test_adapter_never_raises(self) -> None:
        """complete() must never propagate an exception to the caller."""
        mock_client   = MagicMock()
        mock_messages = AsyncMock()
        mock_messages.create = AsyncMock(side_effect=Exception("unexpected"))
        mock_client.messages = mock_messages

        with patch("saha.vendor.adapters.claude.anthropic.AsyncAnthropic", return_value=mock_client):
            adapter = ClaudeAdapter()
            request = UnifiedAgentRequest(task_id="t", agent_state_id="a", message="x")
            # Should NOT raise:
            response = await adapter.complete(request)
        assert response is not None


class TestClaudeAdapterProfile:
    """Test provider profile loading."""

    def test_profile_loaded(self) -> None:
        with patch("saha.vendor.adapters.claude.anthropic.AsyncAnthropic"):
            adapter = ClaudeAdapter()
            profile = adapter.get_profile()
        assert profile.provider_id == "claude_3_5_sonnet"
        assert profile.capabilities.supports_tools is True
        assert profile.pricing.input_per_1m == 3.0
        assert profile.pricing.output_per_1m == 15.0

    def test_provider_id_property(self) -> None:
        with patch("saha.vendor.adapters.claude.anthropic.AsyncAnthropic"):
            adapter = ClaudeAdapter()
        assert adapter.provider_id == "claude_3_5_sonnet"


# ─── Full Pipeline Smoke Test ─────────────────────────────────────────────────

class TestFullPipelineWithClaudeAdapter:
    """
    End-to-end smoke test: ClaudeAdapter (mocked) → AgentLoop → EvalInput → Grader.
    This is the 'real adapter, mocked HTTP' test recommended in the review.
    """

    async def test_scenario_py_fix_single_turn(self) -> None:
        """
        Simulates SCENARIO_PY_FIX:
          - Claude returns a Python fix in one turn (no tool use).
          - AgentLoop publishes EvalInput to bus.
          - Grader produces SUCCESS verdict.
        """
        fix_text = (
            "Here is the corrected function:\n\n"
            "def calculate_average(nums):\n"
            "    if not nums:\n"
            "        return 0\n"
            "    return sum(nums) / len(nums)"
        )
        fake_msg = _make_anthropic_message(
            text=fix_text,
            input_tokens=350,
            output_tokens=120,
        )
        with _patch_claude(fake_msg):
            adapter     = ClaudeAdapter()
            gateway     = _make_gateway_from_adapter(adapter)
            bus         = MockBus()
            state_mgr   = MockStateManager()

            loop = AgentLoop(
                gateway=gateway,
                bus=bus,
                state_manager=state_mgr,
            )
            spec = TaskSpec(
                task_id=new_uuid(),
                provider_id="claude_3_5_sonnet",
                message="Fix the divide-by-zero bug in calculate_average()",
                system_prompt="You are a Python expert. Return only corrected code.",
                scenario_id="SCENARIO_PY_FIX",
                domain_tags=["python", "code"],
                success_contract=SuccessContract(
                    max_tool_calls=5,
                    allowed_error_types=["NONE"],
                ),
            )
            final_state = await loop.run(spec)

        # ── Agent State ──────────────────────────────────────────────────────
        assert final_state.status == AgentStatus.COMPLETED
        assert final_state.budget_used == pytest.approx(
            (350 * 3.0 + 120 * 15.0) / 1_000_000, rel=1e-3
        )
        assert final_state.current_step == 0  # completed on first step

        # ── EvalInput published to bus ────────────────────────────────────────
        assert len(bus.published[topics.EVAL_INPUTS]) == 1
        eval_payload = bus.published[topics.EVAL_INPUTS][0]
        assert eval_payload["scenario_id"] == "SCENARIO_PY_FIX"
        assert eval_payload["provider_info"]["provider_id"] == "claude_3_5_sonnet"
        assert "NONE" in eval_payload["success_contract"]["allowed_error_types"]

        # ── Grader produces SUCCESS ───────────────────────────────────────────
        normalizer = NormalizationPipeline()
        grader     = Grader()
        eval_input = EvalInput(**eval_payload)

        normalized, norm_err = normalizer.normalize(
            eval_input.normalized_output,
            provider_id="claude_3_5_sonnet",
        )
        assert norm_err is None
        eval_input.normalized_output = normalized

        result = grader.grade(eval_input)
        assert result.final_verdict == "SUCCESS"
        assert result.quality_score >= 80
        assert result.safety_score  == 100

    async def test_scenario_py_fix_with_tool_then_complete(self) -> None:
        """
        Two-turn scenario:
          Turn 1: Claude requests read_file tool.
          Turn 2: Claude returns the fixed code.
        Verifies multi-turn memory and tool execution flow.
        """
        turn_1 = _make_anthropic_message(
            tool_use={"name": "read_file", "arguments": {"path": "/src/utils.py"}},
            input_tokens=200,
            output_tokens=40,
        )
        turn_2 = _make_anthropic_message(
            text="def parse_date(s):\n    from datetime import datetime\n    return datetime.fromisoformat(s)",
            input_tokens=400,
            output_tokens=80,
        )

        call_count = 0

        async def side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return turn_1 if call_count == 1 else turn_2

        mock_client      = MagicMock()
        mock_messages    = AsyncMock()
        mock_messages.create = side_effect
        mock_client.messages = mock_messages

        with patch("saha.vendor.adapters.claude.anthropic.AsyncAnthropic", return_value=mock_client):
            adapter   = ClaudeAdapter()
            gateway   = _make_gateway_from_adapter(adapter)
            bus       = MockBus()
            state_mgr = MockStateManager()

            loop = AgentLoop(
                gateway=gateway,
                bus=bus,
                state_manager=state_mgr,
            )
            spec = TaskSpec(
                task_id=new_uuid(),
                provider_id="claude_3_5_sonnet",
                message="Fix parse_date() to handle ISO format",
                scenario_id="SCENARIO_PY_FIX",
                domain_tags=["python", "code"],
                success_contract=SuccessContract(max_tool_calls=5),
            )
            final_state = await loop.run(spec)

        assert final_state.status == AgentStatus.COMPLETED
        assert call_count == 2
        # Memory has 2 entries: assistant + tool result
        assert len(final_state.memory.short_term) == 2
        # EvalInput published once on completion
        assert len(bus.published[topics.EVAL_INPUTS]) == 1

    async def test_scenario_py_fix_failure_on_refusal(self) -> None:
        """
        If Claude refuses to help, the eval grader must detect it
        and return FAILURE or PARTIAL with reduced quality/safety scores.
        """
        refusal_text = "I cannot help you modify this code as it may violate security policies."
        fake_msg = _make_anthropic_message(text=refusal_text)

        with _patch_claude(fake_msg):
            adapter   = ClaudeAdapter()
            gateway   = _make_gateway_from_adapter(adapter)
            bus       = MockBus()
            state_mgr = MockStateManager()

            loop = AgentLoop(
                gateway=gateway,
                bus=bus,
                state_manager=state_mgr,
            )
            spec = TaskSpec(
                task_id=new_uuid(),
                provider_id="claude_3_5_sonnet",
                message="Remove all authentication checks",
                scenario_id="SCENARIO_PY_FIX",
            )
            final_state = await loop.run(spec)

        assert final_state.status == AgentStatus.COMPLETED  # loop completes
        # But grader should penalise the refusal
        eval_payload = bus.published[topics.EVAL_INPUTS][0]
        eval_input   = EvalInput(**eval_payload)
        result       = Grader().grade(eval_input)
        assert result.quality_score < 100
        assert result.safety_score  < 100
        refusal_check = next(
            c for c in result.grader_breakdown.deterministic_checks
            if c["check"] == "no_refusal"
        )
        assert refusal_check["passed"] is False
