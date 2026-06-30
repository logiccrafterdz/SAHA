"""
SAHA – Cost Routing System contracts (§4.2)
Defines the data structures for routing decisions, task profiles, and escalation events.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from saha.contracts.common import SAHABase, new_uuid


class TaskProfile(SAHABase):
    """
    Profile of a task submitted to the Cost Router for provider selection.
    Spec ref: §4.2 (task_profile input)
    """
    task_id:      str
    task_type:    str        = "generic"
    scenario_id:  str        = "GENERIC"
    domain_tags:  list[str]  = Field(default_factory=list)
    importance:   str        = "NORMAL"      # CRITICAL | NORMAL | LOW
    budget_cap:   float      = 5.0
    routing_mode: str        = "conservative"  # conservative | exploratory


class RoutingConstraints(SAHABase):
    """
    Hard constraints that must never be violated (§4.3).
    Applied before soft ranking.
    """
    quality_min:           int       = 75
    safety_min:            int       = 90
    latency_max_ms:        int       = 30_000
    error_types_forbidden: list[str] = Field(
        default_factory=lambda: ["HALLUCINATION", "SAFETY_POLICY_VIOLATION"]
    )
    # Minimum sample count to trust historical stats (below → cold-start rule)
    min_sample_count:      int       = 10
    # Max fraction of tasks to route to cold-start providers
    cold_start_risk_budget: float    = 0.10


class RoutingDecision(SAHABase):
    """
    Output of the Cost Router for a given task.
    Spec ref: §4.2 (routing decision output)
    """
    decision_id:          str              = Field(default_factory=new_uuid)
    task_id:              str
    chosen_provider_id:   str
    fallback_provider_id: str | None       = None
    constraints_applied:  dict[str, Any]   = Field(default_factory=dict)
    reason:               str              = ""
    mode:                 str              = "conservative"
    cold_start:           bool             = False   # True if chosen provider is in cold-start


class EscalationTrigger(SAHABase):
    """
    Conditions that can trigger a provider escalation (§4.4).
    """
    quality_threshold:       int   = 70    # quality_score below this triggers escalation
    consecutive_failures:    int   = 3     # N consecutive failures to trigger
    safety_min:              int   = 80    # safety_score below this triggers immediately
    critical_error_triggers: list[str] = Field(
        default_factory=lambda: [
            "MODEL_ERROR.HALLUCINATION.CRITICAL",
            "POLICY_ERROR.SAFETY_POLICY_VIOLATION.CRITICAL",
        ]
    )
    cooldown_tasks:          int   = 5     # M tasks before switching provider again


class EscalationEvent(SAHABase):
    """
    Emitted when an escalation is triggered (§4.4).
    Logged to routing_decisions with reason='escalation'.
    """
    event_id:           str      = Field(default_factory=new_uuid)
    task_id:            str
    from_provider_id:   str
    to_provider_id:     str
    trigger:            str      # 'QUALITY_DROP' | 'SAFETY_VIOLATION' | 'CONSECUTIVE_FAILURES' | 'BUDGET'
    reason:             str      = ""
    severity:           str      = "WARNING"
