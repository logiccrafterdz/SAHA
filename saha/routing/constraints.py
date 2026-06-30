"""
SAHA – Routing Constraints manager (§4.3)
Manages hard constraints for provider selection.
Supports default constraints, HITL overrides, and per-scenario customization.
"""
from __future__ import annotations

import logging

from saha.contracts.routing import RoutingConstraints

logger = logging.getLogger(__name__)

# ─── Mode-specific defaults ───────────────────────────────────────────────────

_CONSERVATIVE_DEFAULTS = RoutingConstraints(
    quality_min            = 80,
    safety_min             = 90,
    latency_max_ms         = 20_000,
    cold_start_risk_budget = 0.05,   # only 5% of tasks to new providers
)

_EXPLORATORY_DEFAULTS = RoutingConstraints(
    quality_min            = 65,
    safety_min             = 80,
    latency_max_ms         = 45_000,
    cold_start_risk_budget = 0.25,   # allow 25% to new providers
)

# Importance multiplier: CRITICAL tasks get tighter constraints
_IMPORTANCE_OVERRIDES: dict[str, dict] = {
    "CRITICAL": {"quality_min": 90, "safety_min": 95, "cold_start_risk_budget": 0.0},
    "NORMAL":   {},
    "LOW":      {"quality_min": 60, "safety_min": 75},
}


class ConstraintManager:
    """
    Resolves the active RoutingConstraints for a given task profile.
    Priority: HITL override > importance override > routing_mode defaults.
    """

    def __init__(self) -> None:
        # In-memory HITL overrides (keyed by scope: 'global' or 'project_X')
        self._hitl_overrides: dict[str, dict] = {}

    def get_constraints(
        self,
        routing_mode: str,
        importance:   str   = "NORMAL",
        scope:        str   = "global",
    ) -> RoutingConstraints:
        """
        Build effective constraints for a task:
        1. Start from routing_mode defaults.
        2. Apply importance overrides.
        3. Apply HITL overrides (narrowest scope wins).
        """
        base = (
            _CONSERVATIVE_DEFAULTS
            if routing_mode == "conservative"
            else _EXPLORATORY_DEFAULTS
        )
        merged = base.model_copy()

        # Apply importance overrides
        imp_override = _IMPORTANCE_OVERRIDES.get(importance, {})
        for key, val in imp_override.items():
            setattr(merged, key, val)

        # Apply HITL override (most specific scope first)
        for try_scope in [scope, "global"]:
            if try_scope in self._hitl_overrides:
                for key, val in self._hitl_overrides[try_scope].items():
                    if hasattr(merged, key):
                        setattr(merged, key, val)
                logger.debug("HITL override applied for scope=%s", try_scope)
                break

        return merged

    def apply_hitl_override(self, scope: str, change: dict) -> None:
        """Apply a HITL policy override. Persisted in-memory; DB persistence via HITLService."""
        self._hitl_overrides[scope] = {**self._hitl_overrides.get(scope, {}), **change}
        logger.info("ConstraintManager: HITL override applied | scope=%s keys=%s", scope, list(change))

    def clear_override(self, scope: str) -> None:
        self._hitl_overrides.pop(scope, None)

    def validate_against_profile(
        self,
        constraints: RoutingConstraints,
        provider_stats: dict,
    ) -> bool:
        """
        Returns True if a provider's historical stats meet the hard constraints.
        provider_stats: a ProviderWindowStats.to_dict() result.
        """
        if not provider_stats:
            return True  # No data → cold-start (handled separately)

        if provider_stats.get("quality_p50", 0) < constraints.quality_min:
            return False
        if provider_stats.get("safety_avg", 0) < constraints.safety_min:
            return False
        if provider_stats.get("latency_p50_ms", 0) > constraints.latency_max_ms:
            return False
        return True


# Module-level singleton
_constraint_manager = ConstraintManager()


def get_constraint_manager() -> ConstraintManager:
    return _constraint_manager
