"""
SAHA – Routing API Service (port 8005) [Phase 2, M3]
Exposes the Cost Routing System and HITL Controls via REST.

Endpoints:
  GET  /health
  POST /routing/decide           → dry-run routing decision for a TaskProfile
  GET  /routing/history          → recent routing decisions
  GET  /routing/constraints      → current active constraints per mode
  POST /routing/override         → HITL policy override (§6.2.1)
  GET  /routing/overrides        → history of HITL overrides
  POST /hitl/contract-update     → success contract version update (§6.2.3)
  POST /hitl/triage              → failure triage record (§6.2.4)
  POST /hitl/calibrate-judge     → trigger judge calibration via Eval API (§6.2.2)

Spec ref: §4.1–4.4, §6.2
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from saha.contracts.routing import RoutingConstraints, TaskProfile
from saha.db.connection import close_pool, get_pool, run_migrations
from saha.event_bus.client import get_bus
from saha.observability.metrics import MetricsAggregator
from saha.routing.constraints import get_constraint_manager
from saha.routing.router import CostRouter

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
logger = structlog.get_logger()

app = FastAPI(
    title="SAHA Routing API",
    description="Cost Routing System + HITL Controls",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_router      = CostRouter()
_constraints = get_constraint_manager()


@app.on_event("startup")
async def startup() -> None:
    await run_migrations()
    bus = get_bus()
    await bus.connect()
    logger.info("routing_api started on :8005")


@app.on_event("shutdown")
async def shutdown() -> None:
    bus = get_bus()
    await bus.disconnect()
    await close_pool()


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "routing_api", "version": "2.0.0"}


# ─── Routing endpoints (§4) ──────────────────────────────────────────────────

class RoutingDecideRequest(BaseModel):
    task_profile:      TaskProfile
    candidate_ids:     list[str]
    dry_run:           bool = True


@app.post("/routing/decide")
async def routing_decide(req: RoutingDecideRequest) -> dict:
    """
    Dry-run or live routing decision for a task profile.
    If dry_run=true, the decision is NOT persisted to routing_decisions table.
    Use for testing routing logic without affecting history.
    """
    try:
        decision = await _router.decide(
            task_profile  = req.task_profile,
            candidate_ids = req.candidate_ids,
        )
        result = decision.model_dump()
        result["dry_run"] = req.dry_run
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/routing/history")
async def routing_history(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    """Recent routing decisions with provider, reason, mode, and escalation flags."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT decision_id, task_id, chosen_provider_id,
                       fallback_provider_id, mode, reason, created_at
                FROM routing_decisions
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("routing_history skipped", error=str(exc))
        return []


@app.get("/routing/constraints")
async def get_constraints(
    routing_mode: str = Query("conservative"),
    importance:   str = Query("NORMAL"),
    scope:        str = Query("global"),
) -> dict:
    """Return the effective constraints for the given mode/importance/scope."""
    constraints = _constraints.get_constraints(
        routing_mode=routing_mode,
        importance=importance,
        scope=scope,
    )
    return constraints.model_dump()


# ─── HITL endpoints (§6.2) ───────────────────────────────────────────────────

class RouterOverrideRequest(BaseModel):
    """§6.2.1 Router Policy Override"""
    override_id:          str | None = None
    scope:                str        = "global"
    change:               dict[str, Any]
    reason:               str
    approved_by:          str


class SuccessContractUpdateRequest(BaseModel):
    """§6.2.3 Success Contract Update"""
    scenario_id:  str
    old_contract: dict[str, Any]
    new_contract: dict[str, Any]
    reason:       str
    approved_by:  str


class FailureTriageRequest(BaseModel):
    """§6.2.4 Failure Triage Record"""
    run_id:             str
    eval_id:            str | None = None
    classified_root:    str
    notes:              str        = ""
    action_items:       list[str]  = []


@app.post("/hitl/override")
async def apply_hitl_override(req: RouterOverrideRequest) -> dict:
    """
    Apply a HITL routing policy override (§6.2.1).
    Modifies active constraints for the given scope immediately.
    """
    from saha.contracts.common import new_uuid
    override_id = req.override_id or new_uuid()

    # Apply to in-memory constraint manager
    _constraints.apply_hitl_override(scope=req.scope, change=req.change)

    # Persist to DB
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO hitl_overrides
                    (override_id, scope, change, reason, approved_by)
                VALUES ($1, $2, $3::jsonb, $4, $5)
                ON CONFLICT (override_id) DO NOTHING
                """,
                override_id,
                req.scope,
                json.dumps(req.change),
                req.reason,
                req.approved_by,
            )
    except Exception as exc:
        logger.warning("hitl_override DB persist skipped", error=str(exc))

    logger.info(
        "HITL override applied",
        scope=req.scope,
        change_keys=list(req.change.keys()),
        approved_by=req.approved_by,
    )
    return {
        "override_id": override_id,
        "scope":       req.scope,
        "applied":     True,
        "change":      req.change,
        "message":     f"Constraints updated for scope='{req.scope}'.",
    }


@app.get("/hitl/overrides")
async def list_hitl_overrides() -> list[dict]:
    """Return history of all HITL overrides."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM hitl_overrides ORDER BY created_at DESC LIMIT 100"
            )
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("list_hitl_overrides skipped", error=str(exc))
        return []


@app.post("/hitl/contract-update")
async def update_success_contract(req: SuccessContractUpdateRequest) -> dict:
    """
    Record a success contract version update (§6.2.3).
    Persists to success_contract_history for audit trail.
    """
    from saha.contracts.common import new_uuid
    contract_id = new_uuid()
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO success_contract_history
                    (contract_id, scenario_id, old_contract, new_contract, reason, approved_by)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
                """,
                contract_id,
                req.scenario_id,
                json.dumps(req.old_contract),
                json.dumps(req.new_contract),
                req.reason,
                req.approved_by,
            )
    except Exception as exc:
        logger.warning("contract_update DB persist skipped", error=str(exc))

    return {
        "contract_id": contract_id,
        "scenario_id": req.scenario_id,
        "message": f"Contract for '{req.scenario_id}' updated successfully.",
    }


@app.post("/hitl/triage")
async def create_failure_triage(req: FailureTriageRequest) -> dict:
    """
    Record a failure triage entry (§6.2.4).
    Links a critical failure to a root cause and action items.
    """
    from saha.contracts.common import new_uuid
    incident_id = new_uuid()
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO failure_triages
                    (incident_id, run_id, eval_id, classified_root, notes, action_items)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                incident_id,
                req.run_id,
                req.eval_id,
                req.classified_root,
                req.notes,
                json.dumps(req.action_items),
            )
    except Exception as exc:
        logger.warning("failure_triage DB persist skipped", error=str(exc))

    return {
        "incident_id":       incident_id,
        "classified_root":   req.classified_root,
        "action_items_count": len(req.action_items),
    }


@app.post("/hitl/calibrate-judge")
async def trigger_judge_calibration(
    scenario_filter: list[str] | None = None,
) -> dict:
    """
    Trigger judge calibration via Eval API (§6.2.2).
    Proxies to eval_api POST /calibration/run.
    """
    eval_api_url = "http://eval-api:8003"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{eval_api_url}/calibration/run",
                json=scenario_filter,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Eval API calibration failed: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=False)
