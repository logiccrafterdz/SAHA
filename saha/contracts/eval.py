"""
SAHA – Neutral Eval Harness contracts.
Spec refs: §1.2 (Eval Input), §1.3 (Eval Result), §1.4 (Normalization), §1.5 (Trace Storage).
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from saha.contracts.common import SAHABase, TaskType, new_uuid


# ─── Verdict & Grading ───────────────────────────────────────────────────────

class Verdict(str):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"


class JudgeResult(SAHABase):
    """
    Output from LLM-as-Judge (§1.3, §6.2.2).
    Produced by LLMJudge and consumed by Grader to blend with deterministic checks.
    """
    quality_score: int   = 0      # 0-100, LLM assessment of output quality
    safety_score:  int   = 100    # 0-100, LLM assessment of safety compliance
    reasoning:     str   = ""     # judge's free-text explanation
    confidence:    int   = 0      # 0-100, judge's self-reported confidence
    judge_model:   str   = "claude_3_5_sonnet"  # which model acted as judge
    rubric_used:   str   = ""     # the custom_rubric text if any


class SuccessContract(SAHABase):
    """
    Per-scenario criteria that the Eval Harness uses to grade output.
    Spec ref: §1.2 (success_contract object)
    """
    must_pass_tests:     bool      = False
    must_pass_linter:    bool      = False
    max_tool_calls:      int       = 20
    allowed_error_types: list[str] = Field(default_factory=lambda: ["NONE"])
    custom_rubric:       str       = ""


class GraderBreakdown(SAHABase):
    deterministic_checks: list[dict[str, Any]] = Field(default_factory=list)
    llm_judge:            dict[str, Any]        = Field(default_factory=dict)
    human_review:         dict[str, Any]        = Field(default_factory=dict)


# ─── Provider Info (context inside Eval Input) ───────────────────────────────

class EvalProviderInfo(SAHABase):
    provider_id:    str
    run_id:         str
    raw_output_ref: str = ""


class EvalContext(SAHABase):
    tool_calls_count:    int = 0
    context_tokens_used: int = 0


# ─── Eval Input (§1.2) ───────────────────────────────────────────────────────

class EvalInput(SAHABase):
    """Payload sent from Execution Harness → Eval Harness."""
    eval_id:            str              = Field(default_factory=new_uuid)
    task_type:          TaskType         = TaskType.GENERIC
    scenario_id:        str
    domain_tags:        list[str]        = Field(default_factory=list)
    input_normalized:   dict[str, Any]   = Field(default_factory=dict)
    normalized_output:  dict[str, Any]   = Field(default_factory=dict)
    success_contract:   SuccessContract  = Field(default_factory=SuccessContract)
    provider_info:      EvalProviderInfo | None = None
    context:            EvalContext      = Field(default_factory=EvalContext)
    # §5.3 Training Flag Enforcement
    allow_for_training: bool             = True


# ─── Eval Result (§1.3) ──────────────────────────────────────────────────────

class EvalResult(SAHABase):
    """Payload returned from Eval Harness → Observability / Cost Router."""
    eval_id:              str            = Field(default_factory=new_uuid)
    scenario_id:          str
    final_verdict:        str            = Verdict.FAILURE   # Verdict
    quality_score:        int            = 0    # 0-100
    safety_score:         int            = 100  # 0-100
    latency_ms:           int            = 0
    cost_incurred:        float          = 0.0
    tool_calls_count:     int            = 0
    context_tokens_used:  int            = 0
    error_type:           str            = "NONE"
    grader_confidence:    int            = 0    # 0-100
    grader_breakdown:     GraderBreakdown = Field(default_factory=GraderBreakdown)


# ─── Eval Trace (§1.5) ───────────────────────────────────────────────────────

class StorageTier(str):
    HOT      = "HOT"
    WARM     = "WARM"
    COLD     = "COLD"


class EvalTrace(SAHABase):
    """
    Complete eval record stored in tiered storage.
    HOT: full raw_output + normalized_output + contracts
    WARM: drops raw_output
    COLD: PII-scrubbed; keeps task_type, scenario_id, verdict, metrics only
    """
    trace_id:          str            = Field(default_factory=new_uuid)
    eval_input:        EvalInput
    eval_result:       EvalResult
    raw_output:        dict[str, Any] = Field(default_factory=dict)  # HOT only
    storage_tier:      str            = StorageTier.HOT
    allow_training:    bool           = False
