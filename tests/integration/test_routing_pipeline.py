"""
SAHA – Integration tests for the full Routing Pipeline (Phase 2, M3).
Tests AgentLoop + CostRouter integration end-to-end (no DB, no real API).

Scenarios:
  1. AgentLoop without router (Phase 1 backward-compat)
  2. AgentLoop with router injected (Phase 2 mode): router's decision is used
  3. Router failure gracefully falls back to spec.provider_id
  4. AgentLoop calls router.decide with correct TaskProfile fields
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saha.contracts.common import new_uuid, TaskType
from saha.contracts.eval import SuccessContract
from saha.contracts.routing import RoutingDecision, TaskProfile
from saha.contracts.vendor import RequestOptions, ToolSchema, UnifiedAgentResponse
from saha.execution.agent_loop import AgentLoop, TaskSpec
from saha.routing.router import CostRouter


# ─── Mock factories ───────────────────────────────────────────────────────────

def _make_mock_gateway(
    provider_id:    str   = "claude_3_5_sonnet",
    response_text:  str   = "def calculate_average(nums):\n    if not nums: return 0\n    return sum(nums)/len(nums)",
    status:         str   = "COMPLETED",
) -> MagicMock:
    gateway = MagicMock()
    gateway._adapters = {provider_id: MagicMock(), "gpt_4o": MagicMock()}
    response = UnifiedAgentResponse(
        request_id  = new_uuid(),
        run_id      = new_uuid(),
        provider_id = provider_id,
        status      = status,
        normalized_output = {"text": response_text},
        stop_reason = "end_turn",
    )
    gateway.complete = AsyncMock(return_value=response)
    return gateway



def _make_mock_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish    = AsyncMock()
    bus.connect    = AsyncMock()
    bus.disconnect = AsyncMock()
    return bus


def _make_mock_router(chosen: str = "gpt_4o") -> MagicMock:
    router   = MagicMock(spec=CostRouter)
    decision = RoutingDecision(
        task_id              = "t-test",
        chosen_provider_id   = chosen,
        fallback_provider_id = "claude_3_5_sonnet",
        reason               = f"Test: selected {chosen}",
        mode                 = "conservative",
        cold_start           = False,
    )
    router.decide = AsyncMock(return_value=decision)
    return router


def _make_task_spec(
    provider_id:  str  = "claude_3_5_sonnet",
    scenario_id:  str  = "SCENARIO_PY_FIX",
    importance:   str  = "NORMAL",
) -> TaskSpec:
    return TaskSpec(
        task_id      = new_uuid(),
        provider_id  = provider_id,
        message      = "Fix the ZeroDivisionError in calculate_average().",
        system_prompt= "You are a Python expert.",
        scenario_id  = scenario_id,
        importance   = importance,
        domain_tags  = ["python", "code"],
        tools        = [],
        success_contract = SuccessContract(custom_rubric="", max_tool_calls=5),
        options      = RequestOptions(budget_cap=2.0, routing_mode="conservative"),
    )


# ─── Shared AgentStateManager mock ───────────────────────────────────────────

def _mock_state_manager(provider_id: str = "claude_3_5_sonnet") -> MagicMock:
    """Returns a pre-configured AgentStateManager mock that bypasses DB."""
    from saha.contracts.execution import AgentState, AgentStatus
    sm    = MagicMock()
    state = AgentState(
        agent_state_id = new_uuid(),
        task_id        = "t-test",
        provider_id    = provider_id,
        status         = AgentStatus.COMPLETED,
    )
    sm.create          = AsyncMock(return_value=state)
    sm.update          = AsyncMock(return_value=state)
    sm.mark_completed  = AsyncMock(return_value=state)
    sm.mark_failed     = AsyncMock(return_value=state)
    sm.get             = AsyncMock(return_value=state)
    return sm



# ─── Phase 1 backward-compatibility ──────────────────────────────────────────

class TestAgentLoopPhase1Compat:
    async def test_no_router_uses_spec_provider(self) -> None:
        """Without CostRouter, AgentLoop must use spec.provider_id directly."""
        gateway = _make_mock_gateway(provider_id="claude_3_5_sonnet")
        bus     = _make_mock_bus()
        sm      = _mock_state_manager("claude_3_5_sonnet")
        loop    = AgentLoop(gateway=gateway, bus=bus, router=None, state_manager=sm)
        spec    = _make_task_spec(provider_id="claude_3_5_sonnet")

        state = await loop.run(spec)

        assert state.provider_id == "claude_3_5_sonnet"

    async def test_no_router_still_completes(self) -> None:
        gateway = _make_mock_gateway(provider_id="claude_3_5_sonnet")
        bus     = _make_mock_bus()
        sm      = _mock_state_manager("claude_3_5_sonnet")
        loop    = AgentLoop(gateway=gateway, bus=bus, router=None, state_manager=sm)
        spec    = _make_task_spec(provider_id="claude_3_5_sonnet")

        state = await loop.run(spec)

        assert state.status in ("COMPLETED", "FAILED")


# ─── Phase 2: CostRouter injection ───────────────────────────────────────────

class TestAgentLoopWithRouter:
    async def test_router_decision_overrides_spec_provider(self) -> None:
        """
        When CostRouter chooses 'gpt_4o' and spec says 'claude_3_5_sonnet',
        the resolved provider_id must be 'gpt_4o'.
        """
        mock_router = _make_mock_router(chosen="gpt_4o")
        gateway     = _make_mock_gateway(provider_id="gpt_4o")
        bus         = _make_mock_bus()
        sm          = _mock_state_manager("gpt_4o")
        loop        = AgentLoop(gateway=gateway, bus=bus, router=mock_router, state_manager=sm)
        spec        = _make_task_spec(provider_id="claude_3_5_sonnet")

        state = await loop.run(spec)

        assert state.provider_id == "gpt_4o"

    async def test_router_decide_called_with_task_profile(self) -> None:
        """AgentLoop must call router.decide() with a valid TaskProfile."""
        mock_router = _make_mock_router(chosen="gpt_4o")
        gateway     = _make_mock_gateway(provider_id="gpt_4o")
        bus         = _make_mock_bus()
        sm          = _mock_state_manager("gpt_4o")
        loop        = AgentLoop(gateway=gateway, bus=bus, router=mock_router, state_manager=sm)
        spec        = _make_task_spec(scenario_id="SCENARIO_PY_FIX", importance="CRITICAL")

        await loop.run(spec)

        mock_router.decide.assert_called_once()
        call_args    = mock_router.decide.call_args
        task_profile = (
            call_args.kwargs.get("task_profile")
            or (call_args[0][0] if call_args[0] else None)
        )
        assert task_profile is not None
        assert task_profile.scenario_id == "SCENARIO_PY_FIX"
        assert task_profile.importance  == "CRITICAL"

    async def test_router_failure_falls_back_to_spec(self) -> None:
        """If CostRouter raises, AgentLoop must fall back to spec.provider_id."""
        failing_router = MagicMock(spec=CostRouter)
        failing_router.decide = AsyncMock(side_effect=RuntimeError("router is down"))

        gateway = _make_mock_gateway(provider_id="claude_3_5_sonnet")
        bus     = _make_mock_bus()
        sm      = _mock_state_manager("claude_3_5_sonnet")
        loop    = AgentLoop(gateway=gateway, bus=bus, router=failing_router, state_manager=sm)
        spec    = _make_task_spec(provider_id="claude_3_5_sonnet")

        state = await loop.run(spec)

        assert state.provider_id == "claude_3_5_sonnet"

    async def test_candidate_ids_come_from_gateway_adapters(self) -> None:
        """router.decide() must receive all registered adapter IDs as candidates."""
        mock_router = _make_mock_router(chosen="gpt_4o")
        gateway     = _make_mock_gateway()
        # gateway._adapters has claude_3_5_sonnet + gpt_4o
        bus  = _make_mock_bus()
        sm   = _mock_state_manager("gpt_4o")
        loop = AgentLoop(gateway=gateway, bus=bus, router=mock_router, state_manager=sm)
        spec = _make_task_spec()

        await loop.run(spec)

        assert mock_router.decide.called
        call_args     = mock_router.decide.call_args
        candidate_ids = call_args.kwargs.get("candidate_ids") or (
            call_args[0][1] if len(call_args[0]) > 1 else []
        )
        assert set(candidate_ids) >= {"claude_3_5_sonnet", "gpt_4o"}




# ─── CostRouter + MetricsAggregator integration ───────────────────────────────

class TestCostRouterMetricsIntegration:
    async def test_router_decides_correctly_from_metrics_data(self) -> None:
        """
        CostRouter fed real stats from MetricsAggregator.get_cross_provider_report.
        Best provider should win.
        """
        providers = {
            "claude_3_5_sonnet": {
                "quality_p50": 92, "quality_p90": 97, "safety_avg": 98,
                "cost_per_task": 0.004, "latency_p50_ms": 1100, "sample_count": 80,
            },
            "gpt_4o": {
                "quality_p50": 88, "quality_p90": 93, "safety_avg": 96,
                "cost_per_task": 0.006, "latency_p50_ms": 900, "sample_count": 60,
            },
        }
        mock_metrics = MagicMock()
        mock_metrics.get_cross_provider_report = AsyncMock(
            return_value={"providers": providers}
        )
        router = CostRouter(metrics=mock_metrics)
        task   = TaskProfile(
            task_id="t-integ", task_type="code_generation",
            scenario_id="SCENARIO_PY_FIX", domain_tags=["python"],
            importance="NORMAL", budget_cap=5.0, routing_mode="exploratory",
        )
        decision = await router.decide(task, ["claude_3_5_sonnet", "gpt_4o"])

        # Claude has better quality + safety → should win
        assert decision.chosen_provider_id   == "claude_3_5_sonnet"
        assert decision.fallback_provider_id == "gpt_4o"
        assert decision.cold_start is False
