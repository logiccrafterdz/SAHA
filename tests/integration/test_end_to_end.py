"""
SAHA – Integration test: full end-to-end pipeline (mocked provider).
Tests the complete flow: AgentLoop → VendorGateway → EvalInput → Grader
without hitting any external API or database.

Uses:
  - A mock adapter instead of ClaudeAdapter
  - An in-memory AgentStateManager mock
  - A mock Redis bus (collects published messages)
"""
from __future__ import annotations

import pytest
from typing import Any
from collections import defaultdict

from saha.contracts.common import CanonicalError, new_uuid
from saha.contracts.eval import EvalInput, SuccessContract
from saha.contracts.execution import AgentState, AgentStatus
from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderCapabilities,
    ProviderPolicies,
    ProviderPricing,
    ProviderProfile,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)
from saha.eval.grader import Grader
from saha.eval.normalizer import NormalizationPipeline
from saha.event_bus import topics
from saha.execution.agent_loop import AgentLoop, TaskSpec
from saha.vendor.base import BaseAdapter
from saha.vendor import VendorGateway


# ─── Mocks ───────────────────────────────────────────────────────────────────

class MockAdapter(BaseAdapter):
    """Returns a fixed COMPLETED response on every call."""

    def __init__(self, response_text: str = "Task done successfully.") -> None:
        self._text = response_text
        self.call_count = 0

    @property
    def provider_id(self) -> str:
        return "mock_provider"

    async def complete(self, request: UnifiedAgentRequest) -> UnifiedAgentResponse:
        self.call_count += 1
        return UnifiedAgentResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            status="COMPLETED",
            normalized_output={"text": self._text},
            cost_estimate=0.01,
            context_tokens_used=500,
            latency_ms=100,
            error=CanonicalError.none(),
        )

    async def interrupt(self, signal: BudgetInterruptSignal) -> bool:
        return True

    def get_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_id=self.provider_id,
            capabilities=ProviderCapabilities(),
            pricing=ProviderPricing(),
            policies=ProviderPolicies(),
        )


class MockToolAdapter(BaseAdapter):
    """Returns NEEDS_TOOL once, then COMPLETED."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def provider_id(self) -> str:
        return "mock_tool_provider"

    async def complete(self, request: UnifiedAgentRequest) -> UnifiedAgentResponse:
        self.call_count += 1
        if self.call_count == 1:
            return UnifiedAgentResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                status="NEEDS_TOOL",
                normalized_output={"text": "Calling echo tool"},
                pending_tool_call={"name": "echo", "arguments": {"message": "ping"}},
                cost_estimate=0.005,
                latency_ms=80,
                error=CanonicalError.none(),
            )
        return UnifiedAgentResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            status="COMPLETED",
            normalized_output={"text": "Tool result processed. Done."},
            cost_estimate=0.005,
            context_tokens_used=700,
            latency_ms=90,
            error=CanonicalError.none(),
        )

    async def interrupt(self, signal: BudgetInterruptSignal) -> bool:
        return True

    def get_profile(self) -> ProviderProfile:
        return ProviderProfile(provider_id=self.provider_id)


class MockBus:
    """Collects all published messages instead of sending to Redis."""

    def __init__(self) -> None:
        self.published: dict[str, list[dict]] = defaultdict(list)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published[topic].append(payload)

    async def subscribe(self, topic: str, handler: Any) -> None:
        pass  # no-op for tests


class MockStateManager:
    """In-memory agent state store."""

    def __init__(self) -> None:
        self._store: dict[str, AgentState] = {}

    async def create(self, task_id: str, provider_id: str, budget_cap: float = 5.0) -> AgentState:
        state = AgentState(
            task_id=task_id,
            provider_id=provider_id,
            budget_cap=budget_cap,
        )
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


def make_gateway(adapter: BaseAdapter) -> VendorGateway:
    gw = VendorGateway.__new__(VendorGateway)
    gw._adapters = {adapter.provider_id: adapter}
    return gw


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestEndToEndSimpleCompletion:
    async def test_single_turn_completes(self) -> None:
        adapter   = MockAdapter("Hello! Task done.")
        gateway   = make_gateway(adapter)
        bus       = MockBus()
        state_mgr = MockStateManager()

        loop = AgentLoop(
            gateway=gateway,
            bus=bus,
            state_manager=state_mgr,
        )
        spec = TaskSpec(
            task_id=new_uuid(),
            provider_id="mock_provider",
            message="Say hello",
            scenario_id="TEST_HELLO",
        )

        final_state = await loop.run(spec)

        assert final_state.status == AgentStatus.COMPLETED
        assert final_state.budget_used > 0.0
        # EvalInput must have been published
        assert len(bus.published[topics.EVAL_INPUTS]) == 1
        eval_payload = bus.published[topics.EVAL_INPUTS][0]
        assert eval_payload["scenario_id"] == "TEST_HELLO"
        assert adapter.call_count == 1

    async def test_budget_exceeded_stops_loop(self) -> None:
        adapter   = MockAdapter("some output")
        gateway   = make_gateway(adapter)
        bus       = MockBus()
        state_mgr = MockStateManager()

        loop = AgentLoop(
            gateway=gateway,
            bus=bus,
            state_manager=state_mgr,
        )
        from saha.contracts.vendor import RequestOptions
        spec = TaskSpec(
            task_id=new_uuid(),
            provider_id="mock_provider",
            message="Do something",
            # budget_cap=0.0 → budget_used(0.0) >= budget_cap(0.0) → fails immediately
            options=RequestOptions(budget_cap=0.0),
        )

        final_state = await loop.run(spec)
        assert final_state.status == AgentStatus.FAILED
        assert len(bus.published[topics.BUDGET_INTERRUPTS]) >= 1



class TestEndToEndWithToolUse:
    async def test_tool_call_then_complete(self) -> None:
        adapter   = MockToolAdapter()
        gateway   = make_gateway(adapter)
        bus       = MockBus()
        state_mgr = MockStateManager()

        loop = AgentLoop(
            gateway=gateway,
            bus=bus,
            state_manager=state_mgr,
        )
        spec = TaskSpec(
            task_id=new_uuid(),
            provider_id="mock_tool_provider",
            message="Use the echo tool",
            scenario_id="TEST_TOOL",
        )

        final_state = await loop.run(spec)

        assert final_state.status == AgentStatus.COMPLETED
        assert adapter.call_count == 2  # 1 NEEDS_TOOL + 1 COMPLETED
        # Memory must have the tool result
        assert len(final_state.memory.short_term) >= 2


class TestGraderNormalizerPipeline:
    def test_pipeline_produces_success_for_clean_output(self) -> None:
        pipeline = NormalizationPipeline()
        grader   = Grader()

        from saha.contracts.eval import EvalInput, EvalContext
        from saha.contracts.common import TaskType

        raw = {"text": "The solution uses a recursive approach.", "stop_reason": "end_turn"}
        normalized, err = pipeline.normalize(raw, provider_id="mock")
        assert err is None

        eval_input = EvalInput(
            task_type=TaskType.CODE_GENERATION,
            scenario_id="PIPE_TEST",
            normalized_output=normalized,
            context=EvalContext(tool_calls_count=1),
        )
        result = grader.grade(eval_input, latency_ms=200)
        assert result.final_verdict in ("SUCCESS", "PARTIAL")
        assert result.quality_score > 0
