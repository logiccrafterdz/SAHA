"""
SAHA – Eval API Service (port 8003)  [Phase 2]
Exposes the Neutral Eval Harness via REST.
Includes LLM-as-Judge integration and Judge Calibration endpoints (§6.2.2).
Spec ref: §1.1–1.5, §6.2.2
"""
from __future__ import annotations

import logging
import time

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from saha.contracts.eval import EvalInput, EvalResult, EvalTrace
from saha.db.connection import close_pool, run_migrations
from saha.eval.grader import Grader
from saha.eval.judge_calibration import JudgeCalibration
from saha.eval.llm_judge import LLMJudge
from saha.eval.normalizer import NormalizationPipeline
from saha.eval.storage import EvalTraceStorage
from saha.event_bus.client import get_bus
from saha.event_bus import topics

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
logger = structlog.get_logger()

app = FastAPI(
    title="SAHA Eval API",
    description="Neutral Eval Harness – provider-agnostic grading",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_normalizer = NormalizationPipeline()
_grader     = Grader()
_storage    = EvalTraceStorage()
_llm_judge  = LLMJudge()   # Phase 2: LLM-as-Judge singleton
_last_calibration_report: dict | None = None  # cached in-memory for /calibration/status


async def _process_eval(payload: dict) -> None:
    """Handler for SAHA/eval_inputs bus messages."""
    try:
        eval_input = EvalInput(**payload)
        await _run_eval(eval_input)
    except Exception:
        logger.exception("eval_input processing failed", payload_keys=list(payload.keys()))


async def _run_eval(eval_input: EvalInput) -> EvalTrace:
    import time as _time
    start_ms = int(_time.monotonic() * 1000)

    # 1 – Normalise output (§1.4)
    provider_id = (
        eval_input.provider_info.provider_id
        if eval_input.provider_info else ""
    )
    normalized, norm_error = _normalizer.normalize(
        eval_input.normalized_output, provider_id=provider_id
    )
    if norm_error:
        from saha.contracts.eval import EvalResult, Verdict
        result = EvalResult(
            eval_id=eval_input.eval_id,
            scenario_id=eval_input.scenario_id,
            final_verdict=Verdict.FAILURE,
            quality_score=0,
            safety_score=0,
            error_type=f"{norm_error.type}.{norm_error.code}.{norm_error.severity}",
            grader_confidence=0,
        )
    else:
        eval_input.normalized_output = normalized

        # 2 – LLM Judge (Phase 2): run async, graceful fallback on failure (§1.3)
        judge_result = None
        if _llm_judge.is_applicable(eval_input):
            try:
                judge_result = await _llm_judge.judge(eval_input)
            except Exception as exc:
                logger.warning("llm_judge failed, falling back to deterministic", error=str(exc))

        # 3 – Grade (§1.2–1.3)
        latency_ms = int(_time.monotonic() * 1000) - start_ms
        result = _grader.grade(eval_input, latency_ms=latency_ms, judge_result=judge_result)

    # 4 – Persist trace (§1.5)
    trace = await _storage.save(
        eval_input=eval_input,
        eval_result=result,
        raw_output={},
    )

    # 5 – Publish result to SAHA/eval_results
    bus = get_bus()
    await bus.publish(topics.EVAL_RESULTS, result.to_bus_payload())

    logger.info(
        "Eval complete",
        eval_id=eval_input.eval_id,
        verdict=result.final_verdict,
        quality=result.quality_score,
        safety=result.safety_score,
        judge_enabled=result.grader_breakdown.llm_judge.get("enabled", False),
    )
    return trace


@app.on_event("startup")
async def startup() -> None:
    await run_migrations()
    bus = get_bus()
    await bus.connect()
    await bus.subscribe(topics.EVAL_INPUTS, _process_eval)
    logger.info("eval_api started, subscribed to eval_inputs")


@app.on_event("shutdown")
async def shutdown() -> None:
    bus = get_bus()
    await bus.disconnect()
    await close_pool()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "eval_api"}


@app.post("/eval", response_model=EvalResult)
async def run_eval(eval_input: EvalInput) -> EvalResult:
    """
    Synchronously grade an EvalInput payload.
    Returns EvalResult immediately (no bus involved).
    """
    try:
        trace = await _run_eval(eval_input)
        return trace.eval_result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/traces/{trace_id}/promote-warm")
async def promote_warm(trace_id: str) -> dict:
    """Promote an eval trace from HOT to WARM storage tier."""
    await _storage.promote_to_warm(trace_id)
    return {"trace_id": trace_id, "new_tier": "WARM"}


@app.post("/traces/{trace_id}/promote-cold")
async def promote_cold(trace_id: str) -> dict:
    """Promote an eval trace from WARM to COLD storage tier (PII scrubbed)."""
    await _storage.promote_to_cold(trace_id)
    return {"trace_id": trace_id, "new_tier": "COLD"}


# ─── Calibration endpoints (§6.2.2) ─────────────────────────────────────────

@app.get("/calibration/status")
async def calibration_status() -> dict:
    """
    Return the current state of the LLM Judge:
    - whether judge is enabled
    - last calibration report (if any)
    """
    return {
        "judge_configured": True,
        "judge_model":      "claude-3-5-sonnet-20241022",
        "judge_always_on":  False,
        "last_calibration": _last_calibration_report,
    }


@app.post("/calibration/run")
async def run_calibration(
    scenario_filter: list[str] | None = None,
) -> dict:
    """
    Trigger a Judge Calibration run against the Golden Dataset (§6.2.2).
    Optionally filter to specific scenario IDs.
    Returns the CalibrationReport.
    """
    global _last_calibration_report
    try:
        calibrator = JudgeCalibration(judge=_llm_judge)
        report = await calibrator.run(scenario_filter=scenario_filter)
        _last_calibration_report = report.to_dict()
        logger.info(
            "Calibration complete",
            cases_run=report.cases_run,
            mean_deviation=report.overall_mean_deviation,
            judge_enabled=report.judge_enabled,
        )
        return _last_calibration_report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False)
