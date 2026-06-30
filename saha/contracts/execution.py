"""
SAHA – Agent Execution Harness contracts.
Spec refs: §3.2 (Agent State), §3.3 (Execution Loop behavior).
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from saha.contracts.common import SAHABase, new_uuid


class AgentStatus(str):
    RUNNING          = "RUNNING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    COMPLETED        = "COMPLETED"
    FAILED           = "FAILED"


class AgentMemory(SAHABase):
    """Short-term (in-model) + long-term (storage ref) memory."""
    short_term:    list[dict[str, Any]] = Field(default_factory=list)
    long_term_ref: str = ""   # storage://memory/<agent_state_id>


class AgentState(SAHABase):
    """
    Full runtime state for a single agent/task execution.
    Persisted in PostgreSQL (JSONB column) and keyed by agent_state_id.
    Spec ref: §3.2
    """
    agent_state_id:      str       = Field(default_factory=new_uuid)
    task_id:             str
    provider_id:         str
    current_step:        int       = 0
    status:              str       = AgentStatus.RUNNING   # AgentStatus
    memory:              AgentMemory = Field(default_factory=AgentMemory)
    pending_tool_call:   dict[str, Any] | None = None
    budget_used:         float     = 0.0
    budget_cap:          float     = 5.00
    context_tokens_used: int       = 0
