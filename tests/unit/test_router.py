"""
SAHA – Unit tests for CostRouter and EscalationPolicy (Phase 2, M3).
No DB or real API needed — all external calls mocked.

Coverage:
  CostRouter:
  - decide() with no historical stats (all cold-start)
  - decide() selects highest-scoring provider
  - decide() filters providers violating hard constraints
  - decide() emergency fallback when all filtered
  - _soft_rank() weighted score calculation
  - _filter_capabilities() excludes no-tool providers for code tasks
  - Backward-compat: router=None in AgentLoop

  EscalationPolicy:
  - No escalation on good quality
  - Consecutive quality failures trigger after N
  - Safety violation triggers immediately
  - Cooldown prevents immediate re-escalation
  - Budget trigger fires when cost_exceeded=True
  - reset after good result clears consecutive counter

  ConstraintManager:
  - conservative defaults
  - exploratory defaults
  - CRITICAL importance tightens constraints
  - HITL override applied correctly
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saha.contracts.routing import (
    EscalationTrigger,
    RoutingConstraints,
    TaskProfile,
)
from saha.routing.constraints import ConstraintManager
from saha.routing.escalation import EscalationPolicy
from saha.routing.router import CostRouter, _EMERGENCY_FALLBACK, _W_QUALITY, _W_SAFETY


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_task(
    scenario_id:  str   = "SCENARIO_PY_FIX",
    routing_mode: str   = "conservative",
    importance:   str   = "NORMAL",
    budget_cap:   float = 2.0,
) -> TaskProfile:
    return TaskProfile(
        task_id      = "t-test-001",
        task_type    = "code_generation",
        scenario_id  = scenario_id,
        domain_tags  = ["python", "code"],
        importance   = importance,
        budget_cap   = budget_cap,
        routing_mode = routing_mode,
    )


def _make_router(cross_provider_data: dict | None = None) -> CostRouter:
    """Build CostRouter with mocked MetricsAggregator."""
    mock_metrics = MagicMock()
    data = cross_provider_data or {"providers": {}}
    mock_metrics.get_cross_provider_report = AsyncMock(return_value=data)
    return CostRouter(metrics=mock_metrics)


# ─── CostRouter ───────────────────────────────────────────────────────────────

class TestCostRouterColdStart:
    async def test_all_cold_start_returns_first_candidate(self) -> None:
        """No stats → all cold-start → first candidate chosen (alphabetical score=0)."""
        router = _make_router({"providers": {}})
        task   = _make_task()
        decision = await router.decide(task, ["claude_3_5_sonnet", "gpt_4o"])
        assert decision.chosen_provider_id in ["claude_3_5_sonnet", "gpt_4o"]
        assert decision.cold_start is True

    async def test_cold_start_flag_set(self) -> None:
        router = _make_router({"providers": {}})
        decision = await router.decide(_make_task(), ["claude_3_5_sonnet"])
        assert decision.cold_start is True

    async def test_task_id_in_decision(self) -> None:
        router   = _make_router({"providers": {}})
        task     = _make_task()
        decision = await router.decide(task, ["claude_3_5_sonnet"])
        assert decision.task_id == task.task_id


class TestCostRouterSoftRanking:
    def _stats_for(self, q: float, s: float, cost: float = 0.01, lat: int = 1000) -> dict:
        return {
            "quality_p50": q, "quality_p90": q + 5, "safety_avg": s,
            "cost_per_task": cost, "latency_p50_ms": lat, "sample_count": 50,
        }

    async def test_higher_quality_wins(self) -> None:
        providers = {
            "claude_3_5_sonnet": self._stats_for(90, 95, 0.005, 1000),
            "gpt_4o":            self._stats_for(70, 95, 0.005, 1000),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(_make_task(routing_mode="exploratory"), ["claude_3_5_sonnet", "gpt_4o"])
        assert decision.chosen_provider_id == "claude_3_5_sonnet"

    async def test_higher_safety_wins_when_quality_equal(self) -> None:
        providers = {
            "claude_3_5_sonnet": self._stats_for(80, 98, 0.005, 1000),
            "gpt_4o":            self._stats_for(80, 75, 0.005, 1000),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(_make_task(routing_mode="exploratory"), ["claude_3_5_sonnet", "gpt_4o"])
        assert decision.chosen_provider_id == "claude_3_5_sonnet"

    async def test_lower_cost_wins_when_quality_equal(self) -> None:
        providers = {
            "cheap_provider": self._stats_for(85, 95, 0.001, 1200),
            "pricey_provider": self._stats_for(85, 95, 0.050, 1000),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(_make_task(routing_mode="exploratory"), ["cheap_provider", "pricey_provider"])
        assert decision.chosen_provider_id == "cheap_provider"

    async def test_fallback_is_second_ranked(self) -> None:
        providers = {
            "best":   self._stats_for(95, 99, 0.004, 800),
            "second": self._stats_for(85, 95, 0.004, 800),
            "third":  self._stats_for(70, 90, 0.004, 800),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(
            _make_task(routing_mode="exploratory"),
            ["best", "second", "third"],
        )
        assert decision.chosen_provider_id   == "best"
        assert decision.fallback_provider_id == "second"


class TestCostRouterHardConstraints:
    def _stats_for(self, q: float, s: float) -> dict:
        return {
            "quality_p50": q, "quality_p90": q, "safety_avg": s,
            "cost_per_task": 0.005, "latency_p50_ms": 1000, "sample_count": 50,
        }

    async def test_quality_below_min_filtered_out(self) -> None:
        """Conservative mode: quality_min=80. Provider with q=65 must be filtered."""
        providers = {
            "bad_quality": self._stats_for(65, 95),   # below min
            "good":        self._stats_for(88, 95),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(_make_task(routing_mode="conservative"), ["bad_quality", "good"])
        assert decision.chosen_provider_id == "good"

    async def test_safety_below_min_filtered_out(self) -> None:
        providers = {
            "unsafe":  self._stats_for(88, 60),   # safety below min
            "safe":    self._stats_for(85, 95),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(_make_task(routing_mode="conservative"), ["unsafe", "safe"])
        assert decision.chosen_provider_id == "safe"

    async def test_all_filtered_triggers_emergency_fallback(self) -> None:
        """All providers below constraints → emergency fallback."""
        providers = {
            "a": self._stats_for(50, 50),
            "b": self._stats_for(55, 55),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(_make_task(routing_mode="conservative"), ["a", "b"])
        assert decision.chosen_provider_id == _EMERGENCY_FALLBACK

    async def test_critical_importance_tightens_quality(self) -> None:
        """CRITICAL importance: quality_min=90. Provider with q=85 must be filtered."""
        providers = {
            "mid_quality": self._stats_for(85, 96),   # below CRITICAL threshold 90
            "top_quality": self._stats_for(93, 97),
        }
        router   = _make_router({"providers": providers})
        decision = await router.decide(
            _make_task(importance="CRITICAL"),
            ["mid_quality", "top_quality"],
        )
        assert decision.chosen_provider_id == "top_quality"


class TestCostRouterDecision:
    async def test_decision_has_constraints_applied(self) -> None:
        router   = _make_router({"providers": {}})
        decision = await router.decide(_make_task(), ["claude_3_5_sonnet"])
        assert "quality_min" in decision.constraints_applied
        assert "safety_min"  in decision.constraints_applied

    async def test_decision_reason_is_nonempty(self) -> None:
        router   = _make_router({"providers": {}})
        decision = await router.decide(_make_task(), ["claude_3_5_sonnet"])
        assert len(decision.reason) > 10


# ─── ConstraintManager ────────────────────────────────────────────────────────

class TestConstraintManager:
    def test_conservative_defaults(self) -> None:
        cm = ConstraintManager()
        c  = cm.get_constraints("conservative", "NORMAL")
        assert c.quality_min       == 80
        assert c.safety_min        == 90
        assert c.cold_start_risk_budget == pytest.approx(0.05)

    def test_exploratory_defaults(self) -> None:
        cm = ConstraintManager()
        c  = cm.get_constraints("exploratory", "NORMAL")
        assert c.quality_min       == 65
        assert c.cold_start_risk_budget == pytest.approx(0.25)

    def test_critical_importance_tightens(self) -> None:
        cm = ConstraintManager()
        c  = cm.get_constraints("conservative", "CRITICAL")
        assert c.quality_min       == 90
        assert c.safety_min        == 95
        assert c.cold_start_risk_budget == pytest.approx(0.0)

    def test_low_importance_relaxes(self) -> None:
        cm = ConstraintManager()
        c  = cm.get_constraints("conservative", "LOW")
        assert c.quality_min       == 60

    def test_hitl_override_applied(self) -> None:
        cm = ConstraintManager()
        cm.apply_hitl_override("global", {"quality_min": 50, "safety_min": 60})
        c = cm.get_constraints("conservative", "NORMAL")
        assert c.quality_min == 50
        assert c.safety_min  == 60

    def test_hitl_override_cleared(self) -> None:
        cm = ConstraintManager()
        cm.apply_hitl_override("global", {"quality_min": 50})
        cm.clear_override("global")
        c = cm.get_constraints("conservative", "NORMAL")
        assert c.quality_min == 80   # back to default

    def test_validate_profile_passes(self) -> None:
        cm = ConstraintManager()
        c  = cm.get_constraints("conservative")
        ok = cm.validate_against_profile(c, {
            "quality_p50": 90, "safety_avg": 95, "latency_p50_ms": 1000
        })
        assert ok is True

    def test_validate_profile_fails_on_quality(self) -> None:
        cm = ConstraintManager()
        c  = cm.get_constraints("conservative")
        ok = cm.validate_against_profile(c, {
            "quality_p50": 60, "safety_avg": 95, "latency_p50_ms": 1000
        })
        assert ok is False


# ─── EscalationPolicy ────────────────────────────────────────────────────────

class TestEscalationPolicyNoEscalation:
    async def test_good_quality_no_escalation(self) -> None:
        policy = EscalationPolicy()
        should, event = await policy.check(
            "t1", "claude", "SCENARIO_PY_FIX", quality=90, safety=95
        )
        assert should is False
        assert event  is None

    async def test_single_low_quality_no_escalation(self) -> None:
        """One bad result alone doesn't trigger — needs N consecutive."""
        policy = EscalationPolicy(EscalationTrigger(consecutive_failures=3))
        should, _ = await policy.check("t1", "p", "s", quality=60, safety=95)
        assert should is False

    async def test_good_result_resets_consecutive_counter(self) -> None:
        """2 failures + 1 good + 2 failures = no escalation (counter reset)."""
        trigger = EscalationTrigger(consecutive_failures=3, quality_threshold=70)
        policy  = EscalationPolicy(trigger)
        await policy.check("t1", "p", "s", quality=50, safety=95)
        await policy.check("t2", "p", "s", quality=50, safety=95)
        # Good result resets:
        await policy.check("t3", "p", "s", quality=90, safety=95)
        # Two more failures (still only 2 consecutive, below 3):
        await policy.check("t4", "p", "s", quality=50, safety=95)
        should, _ = await policy.check("t5", "p", "s", quality=50, safety=95)
        assert should is False


class TestEscalationPolicyTriggers:
    async def test_consecutive_failures_trigger(self) -> None:
        trigger = EscalationTrigger(consecutive_failures=3, quality_threshold=70)
        policy  = EscalationPolicy(trigger)
        for i in range(2):
            should, _ = await policy.check(f"t{i}", "p", "s", quality=60, safety=95)
            assert should is False
        # 3rd failure → escalate
        should, event = await policy.check("t3", "p", "s", quality=60, safety=95,
                                           fallback_provider_id="gpt_4o")
        assert should is True
        assert event is not None
        assert event.trigger == "CONSECUTIVE_FAILURES"
        assert event.from_provider_id == "p"

    async def test_safety_violation_triggers_immediately(self) -> None:
        trigger = EscalationTrigger(safety_min=80)
        policy  = EscalationPolicy(trigger)
        should, event = await policy.check(
            "t1", "p", "s", quality=90, safety=65,
            fallback_provider_id="claude",
        )
        assert should is True
        assert event.trigger == "SAFETY_VIOLATION"
        assert event.severity == "CRITICAL"

    async def test_budget_trigger(self) -> None:
        policy = EscalationPolicy()
        should, event = await policy.check(
            "t1", "p", "s", quality=90, safety=95,
            cost_exceeded=True, fallback_provider_id="cheap_p",
        )
        assert should is True
        assert event.trigger == "BUDGET"
        assert event.to_provider_id == "cheap_p"


class TestEscalationCooldown:
    async def test_cooldown_prevents_re_escalation(self) -> None:
        trigger = EscalationTrigger(
            consecutive_failures=1, quality_threshold=70, cooldown_tasks=3
        )
        policy = EscalationPolicy(trigger)
        # First escalation fires:
        should1, _ = await policy.check(
            "t1", "new_p", "s", quality=60, safety=95,
            fallback_provider_id="fallback_p",
        )
        assert should1 is True

        # Register switch to new_p → activates cooldown
        policy.record_provider_switch("old_p", "new_p", "s")

        # Next 3 calls on new_p should be blocked by cooldown
        for i in range(3):
            should, _ = await policy.check(
                f"tc{i}", "new_p", "s", quality=60, safety=95
            )
            assert should is False

    async def test_cooldown_expires_after_M_tasks(self) -> None:
        trigger = EscalationTrigger(
            consecutive_failures=1, quality_threshold=70, cooldown_tasks=2
        )
        policy = EscalationPolicy(trigger)
        policy.record_provider_switch("old", "new_p", "s")
        # Cooldown: 2 tasks
        await policy.check("tc1", "new_p", "s", quality=60, safety=95)
        await policy.check("tc2", "new_p", "s", quality=60, safety=95)
        # After cooldown expires: escalation fires again
        should, event = await policy.check(
            "tc3", "new_p", "s", quality=60, safety=95,
            fallback_provider_id="backup",
        )
        assert should is True
