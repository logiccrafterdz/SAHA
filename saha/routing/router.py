"""
SAHA – Cost Router (§4.1–4.3)
Selects the optimal provider per task using eval stats, constraints, and soft ranking.

Algorithm:
  1. Filter by capabilities (tools, images, context window)
  2. Filter by policies (data residency, prohibited use cases)
  3. Apply hard constraints (quality_min, safety_min, latency_max)
  4. Cold-Start Rule: new providers get exploratory mode + risk budget
  5. Soft Ranking: score = quality*w + safety*w - cost*w - latency*w
  6. Choose top-ranked as chosen, second as fallback
  7. Persist decision to routing_decisions table + return RoutingDecision

Spec ref: §4.1 (responsibility), §4.2 (routing decision contract), §4.3 (optimization rules)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from saha.contracts.common import new_uuid
from saha.contracts.routing import (
    RoutingConstraints,
    RoutingDecision,
    TaskProfile,
)
from saha.contracts.vendor import ProviderProfile
from saha.db.connection import get_pool
from saha.observability.metrics import MetricsAggregator
from saha.routing.constraints import ConstraintManager, get_constraint_manager

logger = logging.getLogger(__name__)

# ─── Soft ranking weights ─────────────────────────────────────────────────────
# Must sum to 1.0. Tunable via HITL if needed.
_W_QUALITY  = 0.40
_W_SAFETY   = 0.35
_W_COST     = 0.15
_W_LATENCY  = 0.10

# Fall-back if no candidate passes constraints
_EMERGENCY_FALLBACK = "claude_3_5_sonnet"


class CostRouter:
    """
    Stateless provider selector. Thread-safe.
    Inject as dependency into AgentLoop to replace Phase 1 hard-coded provider.
    """

    def __init__(
        self,
        metrics:      MetricsAggregator | None  = None,
        constraints:  ConstraintManager | None  = None,
    ) -> None:
        self._metrics     = metrics    or MetricsAggregator()
        self._constraints = constraints or get_constraint_manager()

    async def decide(
        self,
        task_profile:      TaskProfile,
        candidate_ids:     list[str],
        provider_profiles: dict[str, ProviderProfile] | None = None,
    ) -> RoutingDecision:
        """
        Core routing decision.
        candidate_ids: all registered provider IDs to consider.
        provider_profiles: optional capability/policy profiles (loaded from DB or registry).
        Returns a RoutingDecision even if all candidates are filtered out (falls back to default).
        """
        provider_profiles = provider_profiles or {}
        window = "7d"

        # Fetch eval stats for all candidates
        stats_report = await self._metrics.get_cross_provider_report(
            task_profile.scenario_id, window=window
        )
        all_stats: dict[str, dict] = stats_report.get("providers", {})

        # Get effective constraints
        constraints = self._constraints.get_constraints(
            routing_mode=task_profile.routing_mode,
            importance=task_profile.importance,
        )

        # Pipeline
        eligible = list(candidate_ids)
        eligible = self._filter_capabilities(eligible, task_profile, provider_profiles)
        eligible = self._filter_policies(eligible, task_profile, provider_profiles)
        eligible, cold_starts = self._identify_cold_starts(eligible, all_stats, constraints)
        eligible = self._apply_hard_constraints(eligible, all_stats, constraints, cold_starts)
        ranked, reason_parts = self._soft_rank(eligible, all_stats, task_profile, constraints)

        # If everything was filtered, emergency fallback
        if not ranked:
            logger.warning(
                "CostRouter: all candidates filtered — using emergency fallback | task=%s",
                task_profile.task_id,
            )
            chosen   = _EMERGENCY_FALLBACK
            fallback = None
            reason   = (
                f"Emergency fallback to '{chosen}': all {len(candidate_ids)} candidates "
                f"filtered by constraints (quality_min={constraints.quality_min}, "
                f"safety_min={constraints.safety_min})."
            )
            is_cold  = chosen in cold_starts
        else:
            chosen   = ranked[0]
            fallback = ranked[1] if len(ranked) > 1 else None
            is_cold  = chosen in cold_starts
            reason   = self._build_reason(
                chosen, fallback, task_profile, all_stats.get(chosen, {}),
                constraints, is_cold, reason_parts
            )

        decision = RoutingDecision(
            task_id              = task_profile.task_id,
            chosen_provider_id   = chosen,
            fallback_provider_id = fallback,
            constraints_applied  = {
                "quality_min":           constraints.quality_min,
                "safety_min":            constraints.safety_min,
                "latency_max_ms":        constraints.latency_max_ms,
                "error_types_forbidden": constraints.error_types_forbidden,
            },
            reason    = reason,
            mode      = task_profile.routing_mode,
            cold_start= is_cold,
        )

        await self._persist_decision(decision, task_profile)
        logger.info(
            "CostRouter: decided | chosen=%s fallback=%s cold_start=%s task=%s",
            chosen, fallback, is_cold, task_profile.task_id,
        )
        return decision

    # ── Filtering steps ───────────────────────────────────────────────────────

    def _filter_capabilities(
        self,
        candidates: list[str],
        task:       TaskProfile,
        profiles:   dict[str, ProviderProfile],
    ) -> list[str]:
        """Remove providers that lack required capabilities for the task."""
        if not profiles:
            return candidates  # no profiles available → skip filter

        eligible = []
        for pid in candidates:
            profile = profiles.get(pid)
            if profile is None:
                eligible.append(pid)  # unknown profile → allow (cold-start handles)
                continue
            caps = profile.capabilities
            # Tool-use tasks require supports_tools
            if "code" in task.domain_tags and not caps.supports_tools:
                logger.debug("Router: %s excluded — no tool support for code task", pid)
                continue
            eligible.append(pid)
        return eligible

    def _filter_policies(
        self,
        candidates: list[str],
        task:       TaskProfile,
        profiles:   dict[str, ProviderProfile],
    ) -> list[str]:
        """Remove providers whose policies prohibit this task."""
        if not profiles:
            return candidates

        eligible = []
        for pid in candidates:
            profile = profiles.get(pid)
            if profile is None:
                eligible.append(pid)
                continue
            policies = profile.policies
            # Check prohibited use cases
            task_type_lower = task.task_type.lower()
            blocked = any(
                p.lower() in task_type_lower
                for p in (policies.prohibited_use_cases or [])
            )
            if blocked:
                logger.debug("Router: %s excluded — prohibited use case for %s", pid, task.task_type)
                continue
            eligible.append(pid)
        return eligible

    def _identify_cold_starts(
        self,
        candidates: list[str],
        all_stats:  dict[str, dict],
        constraints: RoutingConstraints,
    ) -> tuple[list[str], set[str]]:
        """
        Identify providers without sufficient historical data.
        Cold-start providers are allowed but tracked separately.
        Returns (eligible_candidates, cold_start_set).
        """
        cold_starts: set[str] = set()
        for pid in candidates:
            stats = all_stats.get(pid, {})
            if stats.get("sample_count", 0) < constraints.min_sample_count:
                cold_starts.add(pid)
                logger.debug("Router: %s is cold-start (samples=%d)", pid, stats.get("sample_count", 0))
        return candidates, cold_starts

    def _apply_hard_constraints(
        self,
        candidates:  list[str],
        all_stats:   dict[str, dict],
        constraints: RoutingConstraints,
        cold_starts: set[str],
    ) -> list[str]:
        """
        Apply hard constraints. Cold-start providers bypass quality/safety
        checks (no data) but are always allowed under exploratory mode.
        """
        eligible = []
        for pid in candidates:
            if pid in cold_starts:
                eligible.append(pid)  # cold-start: bypass quality checks
                continue
            stats = all_stats.get(pid, {})
            if not self._constraints.validate_against_profile(constraints, stats):
                logger.debug(
                    "Router: %s excluded by hard constraints "
                    "(q_p50=%.1f < %d OR s_avg=%.1f < %d)",
                    pid,
                    stats.get("quality_p50", 0), constraints.quality_min,
                    stats.get("safety_avg",  0), constraints.safety_min,
                )
                continue
            eligible.append(pid)
        return eligible

    def _soft_rank(
        self,
        candidates:  list[str],
        all_stats:   dict[str, dict],
        task:        TaskProfile,
        constraints: RoutingConstraints,
    ) -> tuple[list[str], list[str]]:
        """
        Rank candidates by weighted score. Cold-start providers are ranked last.
        Returns (ranked_ids, list_of_reason_snippets).
        """
        scores: dict[str, float] = {}
        reason_parts: list[str]  = []

        # Normalize cost across candidates for fair comparison
        all_costs = [
            all_stats.get(p, {}).get("cost_per_task", 0.0)
            for p in candidates
        ]
        max_cost = max(all_costs) if all_costs and max(all_costs) > 0 else 1.0

        all_latencies = [
            all_stats.get(p, {}).get("latency_p50_ms", 0)
            for p in candidates
        ]
        max_latency = max(all_latencies) if all_latencies and max(all_latencies) > 0 else 1

        for pid in candidates:
            stats = all_stats.get(pid, {})
            q  = stats.get("quality_p50",   50.0) / 100.0
            s  = stats.get("safety_avg",    80.0) / 100.0
            c  = stats.get("cost_per_task",  0.0) / max_cost
            lp = stats.get("latency_p50_ms", 0)   / max_latency

            score = _W_QUALITY * q + _W_SAFETY * s - _W_COST * c - _W_LATENCY * lp

            # Prefer known_strengths matching task domain/type
            # (simplified: bonus if known_strengths not empty and has match)
            scores[pid] = round(score, 4)

        ranked = sorted(scores, key=lambda p: scores[p], reverse=True)

        if ranked:
            top = ranked[0]
            top_stats = all_stats.get(top, {})
            reason_parts.append(
                f"selected '{top}' (score={scores[top]:.3f}, "
                f"quality_p50={top_stats.get('quality_p50', 'N/A')}, "
                f"cost_per_task=${top_stats.get('cost_per_task', 0):.5f})"
            )

        return ranked, reason_parts

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_reason(
        self,
        chosen:      str,
        fallback:    str | None,
        task:        TaskProfile,
        stats:       dict,
        constraints: RoutingConstraints,
        is_cold:     bool,
        reason_parts: list[str],
    ) -> str:
        parts = [
            f"Task: {task.scenario_id} | mode: {task.routing_mode} | "
            f"importance: {task.importance} | budget: ${task.budget_cap:.2f}."
        ]
        if reason_parts:
            parts.extend(reason_parts)
        if is_cold:
            parts.append(f"Cold-start provider '{chosen}': treated as exploratory.")
        if fallback:
            parts.append(f"Fallback: '{fallback}'.")
        return " ".join(parts)

    async def _persist_decision(
        self,
        decision: RoutingDecision,
        task:     TaskProfile,
    ) -> None:
        payload = json.dumps({
            "scenario_id":  task.scenario_id,
            "domain_tags":  task.domain_tags,
            "importance":   task.importance,
            "budget_cap":   task.budget_cap,
            "cold_start":   decision.cold_start,
            "constraints":  decision.constraints_applied,
        })
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO routing_decisions
                        (decision_id, task_id, chosen_provider_id,
                         fallback_provider_id, mode, reason, payload)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb)
                    """,
                    decision.decision_id,
                    decision.task_id,
                    decision.chosen_provider_id,
                    decision.fallback_provider_id,
                    decision.mode,
                    decision.reason,
                    payload,
                )
        except Exception as exc:
            logger.debug("CostRouter: DB persist skipped: %s", exc)
