"""
SAHA – Vendor Abstraction Layer contracts.
Spec refs: §2.2 (Unified Agent Request), §2.3 (Unified Agent Response),
           §2.4 (Provider Capability & Policy Profile).
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from saha.contracts.common import (
    CanonicalError,
    HarnessPattern,
    RoutingMode,
    SAHABase,
    new_uuid,
)


# ─── Tool Schema ─────────────────────────────────────────────────────────────

class ToolSchema(SAHABase):
    """A single tool exposed to the provider."""
    name:         str
    input_schema: dict[str, Any] = Field(default_factory=dict)


# ─── Request Options ─────────────────────────────────────────────────────────

class RequestOptions(SAHABase):
    max_context_tokens:         int            = 200_000
    enable_arena_mode:          bool           = False
    enable_context_circulation: bool           = False
    harness_pattern:            HarnessPattern = HarnessPattern.SINGLE_AGENT
    parallel_agents:            int            = 1
    budget_cap:                 float          = 5.00
    routing_mode:               RoutingMode    = RoutingMode.CONSERVATIVE


# ─── Unified Agent Request (§2.2) ────────────────────────────────────────────

class UnifiedAgentRequest(SAHABase):
    """Canonical request sent from Execution Harness → Vendor Abstraction."""
    request_id:     str            = Field(default_factory=new_uuid)
    task_id:        str
    agent_state_id: str
    message:        str
    system_prompt:  str            = ""
    tools:          list[ToolSchema] = Field(default_factory=list)
    options:        RequestOptions = Field(default_factory=RequestOptions)


# ─── Unified Agent Response (§2.3) ───────────────────────────────────────────

class ResponseStatus(str):
    COMPLETED  = "COMPLETED"
    NEEDS_TOOL = "NEEDS_TOOL"
    FAILED     = "FAILED"


class UnifiedAgentResponse(SAHABase):
    """Canonical response returned from Vendor Abstraction → Execution Harness."""
    request_id:           str
    provider_id:          str
    run_id:               str = Field(default_factory=new_uuid)
    status:               str  # COMPLETED | NEEDS_TOOL | FAILED
    normalized_output:    dict[str, Any] = Field(default_factory=dict)
    raw_output_ref:       str = ""   # e.g. "storage://hot/..."
    tool_calls_count:     int = 0
    context_tokens_used:  int = 0
    cost_estimate:        float = 0.0
    latency_ms:           int = 0
    error:                CanonicalError = Field(default_factory=CanonicalError.none)

    # When status == NEEDS_TOOL, this holds the call details
    pending_tool_call: dict[str, Any] | None = None


# ─── Provider Capabilities ───────────────────────────────────────────────────

class ProviderCapabilities(SAHABase):
    max_context_tokens:           int  = 200_000
    supports_tools:               bool = True
    supports_images:              bool = False
    supports_parallel_agents:     bool = False
    max_parallel_agents:          int  = 1
    supports_arena_mode:          bool = False
    supports_context_circulation: bool = False
    supports_3_agent_harness:     bool = False
    native_multi_turn:            bool = True
    streaming:                    bool = True
    supports_streaming_cost_tracking: bool = False
    supports_budget_interrupt:    bool = True


class ProviderPricing(SAHABase):
    input_per_1m:  float = 0.0
    output_per_1m: float = 0.0


class ProviderPolicies(SAHABase):
    can_route_to_competitors:       bool       = True
    can_store_outputs_for_training: bool       = False
    can_use_in_eval_comparison:     bool       = True
    data_residency_requirements:    list[str]  = Field(default_factory=list)
    prohibited_use_cases:           list[str]  = Field(default_factory=list)


# ─── Provider Capability & Policy Profile (§2.4) ─────────────────────────────

class ProviderProfile(SAHABase):
    """Full profile for a provider, stored in DB and updated by Observability."""
    provider_id:       str
    capabilities:      ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    pricing:           ProviderPricing      = Field(default_factory=ProviderPricing)
    known_strengths:   list[str]            = Field(default_factory=list)
    known_weaknesses:  list[str]            = Field(default_factory=list)
    policies:          ProviderPolicies     = Field(default_factory=ProviderPolicies)


# ─── Budget Interrupt Signal (§3.4) ──────────────────────────────────────────

class BudgetInterruptSignal(SAHABase):
    """Emitted by Execution Harness when budget_cap is about to be exceeded."""
    command_id:  str   = Field(default_factory=new_uuid)
    task_id:     str
    run_id:      str
    provider_id: str
    reason:      str   = "BUDGET_CAP_REACHED"
    budget_cap:  float
    budget_used: float
