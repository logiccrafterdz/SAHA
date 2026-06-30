"""
SAHA – Observability API Service (port 8004) [Phase 2, M2]
Exposes provider metrics, anomaly log, routing history, and routing decisions.
Consumers: Cost Router (§4.3), HITL dashboards, operational teams.

Endpoints:
  GET /health
  GET /metrics/providers              → all providers summary (window param)
  GET /metrics/providers/{id}         → single provider full stats
  GET /metrics/providers/{id}/profile → provider profile + stats combined
  GET /metrics/scenarios/{id}         → cross-provider comparison for scenario
  GET /anomalies                      → recent unresolved anomaly log
  POST /anomalies/check               → trigger anomaly detection run
  GET /routing/history                → recent routing decisions with reasons
  POST /metrics/refresh               → recompute all provider_stats

Spec ref: §5.1 (Observability responsibility), §5.2 (trace contracts)
"""
from __future__ import annotations

import logging

import structlog
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from saha.db.connection import close_pool, run_migrations
from saha.event_bus.client import get_bus
from saha.observability.anomaly_detector import AnomalyDetector
from saha.observability.metrics import MetricsAggregator

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
logger = structlog.get_logger()

app = FastAPI(
    title="SAHA Observability API",
    description="Provider metrics, anomaly detection, and routing history",
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

_aggregator = MetricsAggregator()
_detector   = AnomalyDetector()


@app.on_event("startup")
async def startup() -> None:
    await run_migrations()
    bus = get_bus()
    await bus.connect()
    logger.info("observability_api started on :8004")


@app.on_event("shutdown")
async def shutdown() -> None:
    bus = get_bus()
    await bus.disconnect()
    await close_pool()


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "observability_api", "version": "2.0.0"}


# ─── Metrics endpoints ────────────────────────────────────────────────────────

@app.get("/metrics/providers")
async def list_provider_metrics(
    window: str = Query("7d", description="Time window: 24h | 7d | 30d"),
) -> list[dict]:
    """
    Return aggregated performance stats for all providers in the given window.
    Sorted by quality_p50 desc. Used by Cost Router as recent_eval_stats input.
    """
    return await _aggregator.get_all_providers_summary(window=window)


@app.get("/metrics/providers/{provider_id}")
async def get_provider_metrics(
    provider_id: str,
    window: str = Query("7d", description="Time window: 24h | 7d | 30d"),
) -> list[dict]:
    """Return all scenario stats for a specific provider."""
    return await _aggregator.get_provider_stats(provider_id=provider_id, window=window)


@app.get("/metrics/scenarios/{scenario_id}")
async def get_scenario_metrics(
    scenario_id: str,
    window: str = Query("7d", description="Time window: 24h | 7d | 30d"),
) -> dict:
    """
    Cross-provider comparison for a scenario.
    Example: compare Claude vs GPT-4o vs Gemini on SCENARIO_PY_FIX.
    """
    return await _aggregator.get_cross_provider_report(
        scenario_id=scenario_id, window=window
    )


@app.post("/metrics/refresh")
async def refresh_metrics() -> dict:
    """
    Recompute all provider_stats from raw eval_traces.
    Call this after bulk data imports or on a scheduled basis.
    """
    stats = await _aggregator.compute_all()
    return {
        "status":    "refreshed",
        "entries":   len(stats),
        "message":   f"Computed {len(stats)} provider×scenario×window entries.",
    }


# ─── Anomaly endpoints ───────────────────────────────────────────────────────

@app.get("/anomalies")
async def list_anomalies(limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    """Return recent unresolved anomalies from anomaly_log."""
    return await _detector.get_recent_anomalies(limit=limit)


@app.post("/anomalies/check")
async def run_anomaly_check() -> dict:
    """
    Trigger an immediate anomaly detection run.
    Detected anomalies are persisted to anomaly_log and published to bus.
    """
    anomalies = await _detector.run_all_checks()
    return {
        "anomalies_found": len(anomalies),
        "types":           [a.type for a in anomalies],
    }


# ─── Routing history ─────────────────────────────────────────────────────────

@app.get("/routing/history")
async def routing_history(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    """
    Return recent routing decisions with chosen provider, reason, and mode.
    Used for HITL review and thrashing detection.
    """
    from saha.db.connection import get_pool
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
        logger.warning("routing_history skipped (no DB?)", error=str(exc))
        return []


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=False)
