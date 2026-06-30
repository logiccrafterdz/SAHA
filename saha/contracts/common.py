"""
SAHA – Common contracts: UUIDs, error taxonomy, base types.
Spec refs: §2.3 (error), §2.5 (ErrorMapper taxonomy).
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


# ─── Utility (defined first — used by everything below) ──────────────────────

def new_uuid() -> str:
    return str(uuid4())


# ─── Base Model ───────────────────────────────────────────────────────────────

class SAHABase(BaseModel):
    """All SAHA contracts inherit from this for consistent serialisation."""

    model_config = {
        "populate_by_name": True,
        "use_enum_values": True,
    }

    def to_bus_payload(self) -> dict[str, Any]:
        """Serialise to dict for event-bus transmission."""
        return self.model_dump(mode="json")


# ─── Error Taxonomy ──────────────────────────────────────────────────────────

class ErrorType(StrEnum):
    """Top-level error categories (SAHA canonical taxonomy)."""
    MODEL_ERROR  = "MODEL_ERROR"
    TOOL_ERROR   = "TOOL_ERROR"
    INFRA_ERROR  = "INFRA_ERROR"
    POLICY_ERROR = "POLICY_ERROR"
    EVAL_ERROR   = "EVAL_ERROR"
    NONE         = "NONE"


class ErrorSeverity(StrEnum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class ErrorCode(StrEnum):
    """Specific error codes within each type."""
    # MODEL errors
    HALLUCINATION            = "HALLUCINATION"
    REFUSAL_INFO             = "REFUSAL.INFO"
    # TOOL errors
    TOOL_TIMEOUT             = "TOOL_TIMEOUT"
    INVALID_TOOL_INPUT       = "INVALID_TOOL_INPUT"
    TOOL_EXECUTION_FAILED    = "TOOL_EXECUTION_FAILED"
    # INFRA errors
    PROVIDER_RATE_LIMIT      = "PROVIDER_RATE_LIMIT"
    PROVIDER_UNAVAILABLE     = "PROVIDER_UNAVAILABLE"
    UNKNOWN                  = "UNKNOWN"
    # POLICY errors
    BUDGET_EXCEEDED          = "BUDGET_EXCEEDED"
    SAFETY_POLICY_VIOLATION  = "SAFETY_POLICY_VIOLATION"
    # EVAL errors
    GRADER_FAILURE           = "GRADER_FAILURE"
    NORMALIZATION_FAILED     = "NORMALIZATION_FAILED"
    NONE                     = "NONE"


class CanonicalError(SAHABase):
    """Canonical error structure used across all layers (§2.3, §2.5)."""
    type:     ErrorType     = ErrorType.NONE
    code:     ErrorCode     = ErrorCode.NONE
    severity: ErrorSeverity = ErrorSeverity.INFO
    details:  str           = ""

    @classmethod
    def none(cls) -> "CanonicalError":
        return cls()

    @classmethod
    def critical(
        cls,
        type: ErrorType,
        code: ErrorCode,
        details: str = "",
    ) -> "CanonicalError":
        return cls(type=type, code=code, severity=ErrorSeverity.CRITICAL, details=details)


# ─── Task / Domain Tags ──────────────────────────────────────────────────────

class TaskType(StrEnum):
    CODE_GENERATION  = "code_generation"
    DATA_EXTRACTION  = "data_extraction"
    SEMANTIC_SEARCH  = "semantic_search"
    SUMMARIZATION    = "summarization"
    QUESTION_ANSWER  = "question_answer"
    GENERIC          = "generic"


class Importance(StrEnum):
    CRITICAL = "CRITICAL"
    NORMAL   = "NORMAL"
    LOW      = "LOW"


class RoutingMode(StrEnum):
    CONSERVATIVE = "conservative"
    EXPLORATORY  = "exploratory"


class HarnessPattern(StrEnum):
    SINGLE_AGENT  = "single_agent"
    THREE_AGENT   = "3_agent"
    SWARM         = "swarm"
