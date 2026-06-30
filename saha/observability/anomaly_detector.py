"""
SAHA – Anomaly Detector (§5.1)
Monitors provider_stats and execution traces for anomalies.
Publishes alerts to SAHA/anomaly_alerts bus topic.

Detected anomaly types:
  QUALITY_DROP      — quality_p50 dropped > 10pp vs previous window
  COST_SPIKE        — cost_per_task increased > 50% vs previous window
  SAFETY_VIOLATION  — any eval_trace with safety_score < SAFETY_FLOOR
  THRASHING         — > N provider changes in 24h (routing instability)

Spec ref: §5.1 (anomaly detection), §5.2 (trace contracts)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from saha.contracts.common import new_uuid
from saha.db.connection import get_pool
from saha.event_bus.client import SAHABusClient, get_bus
from saha.event_bus import topics

logger = logging.getLogger(__name__)

# ─── Thresholds ───────────────────────────────────────────────────────────────

QUALITY_DROP_THRESHOLD = 10.0    # pp drop from 7d → 24h to trigger alert
COST_SPIKE_THRESHOLD   = 0.50    # 50% increase
SAFETY_FLOOR           = 70      # safety_score below this is always anomalous
THRASH_MAX_CHANGES     = 3       # max provider changes in 24h before thrashing alert


@dataclass
class Anomaly:
    anomaly_id:  str
    type:        str
    provider_id: str | None
    scenario_id: str | None
    severity:    str
    details:     dict[str, Any]
    created_at:  str

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id":  self.anomaly_id,
            "type":        self.type,
            "provider_id": self.provider_id,
            "scenario_id": self.scenario_id,
            "severity":    self.severity,
            "details":     self.details,
            "created_at":  self.created_at,
        }


class AnomalyDetector:
    """
    Runs anomaly checks and publishes alerts to the event bus.
    Should be called periodically (e.g., every hour by the observability service).
    """

    def __init__(self, bus: SAHABusClient | None = None) -> None:
        self._bus = bus or get_bus()

    async def run_all_checks(self) -> list[Anomaly]:
        """Run all checks and return detected anomalies."""
        anomalies: list[Anomaly] = []
        try:
            anomalies += await self._check_quality_drops()
            anomalies += await self._check_cost_spikes()
            anomalies += await self._check_safety_violations()
            anomalies += await self._check_thrashing()
        except Exception as exc:
            logger.warning("AnomalyDetector run skipped (no DB?): %s", exc)
            return []

        for anomaly in anomalies:
            await self._persist(anomaly)
            await self._bus.publish(topics.ANOMALY_ALERTS, anomaly.to_dict())
            logger.warning(
                "Anomaly detected | type=%s provider=%s severity=%s",
                anomaly.type, anomaly.provider_id, anomaly.severity,
            )

        return anomalies

    async def get_recent_anomalies(self, limit: int = 50) -> list[dict]:
        """Fetch recent anomalies from DB for the Observability API."""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM anomaly_log
                    WHERE resolved = FALSE
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("get_recent_anomalies skipped: %s", exc)
            return []

    # ── Checks ────────────────────────────────────────────────────────────────

    async def _check_quality_drops(self) -> list[Anomaly]:
        """Detect quality_p50 drops > QUALITY_DROP_THRESHOLD between 7d → 24h."""
        anomalies: list[Anomaly] = []
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p24.provider_id, p24.scenario_id,
                       p24.quality_p50 AS q_24h,
                       p7d.quality_p50 AS q_7d
                FROM provider_stats p24
                JOIN provider_stats p7d
                  ON p24.provider_id = p7d.provider_id
                 AND p24.scenario_id = p7d.scenario_id
                WHERE p24.window = '24h' AND p7d.window = '7d'
                  AND (p7d.quality_p50 - p24.quality_p50) > $1
                  AND p7d.sample_count >= 5
                """,
                QUALITY_DROP_THRESHOLD,
            )
            for row in rows:
                drop = row["q_7d"] - row["q_24h"]
                anomalies.append(Anomaly(
                    anomaly_id  = new_uuid(),
                    type        = "QUALITY_DROP",
                    provider_id = row["provider_id"],
                    scenario_id = row["scenario_id"],
                    severity    = "WARNING" if drop < 20 else "CRITICAL",
                    details     = {
                        "quality_24h": round(row["q_24h"], 1),
                        "quality_7d":  round(row["q_7d"], 1),
                        "drop_pp":     round(drop, 1),
                        "threshold":   QUALITY_DROP_THRESHOLD,
                    },
                    created_at  = datetime.now(timezone.utc).isoformat(),
                ))
        return anomalies

    async def _check_cost_spikes(self) -> list[Anomaly]:
        """Detect cost_per_task increases > COST_SPIKE_THRESHOLD between 7d → 24h."""
        anomalies: list[Anomaly] = []
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p24.provider_id, p24.scenario_id,
                       p24.cost_per_task AS c_24h,
                       p7d.cost_per_task AS c_7d
                FROM provider_stats p24
                JOIN provider_stats p7d
                  ON p24.provider_id = p7d.provider_id
                 AND p24.scenario_id = p7d.scenario_id
                WHERE p24.window = '24h' AND p7d.window = '7d'
                  AND p7d.cost_per_task > 0
                  AND (p24.cost_per_task - p7d.cost_per_task) / p7d.cost_per_task > $1
                """,
                COST_SPIKE_THRESHOLD,
            )
            for row in rows:
                pct = ((row["c_24h"] - row["c_7d"]) / row["c_7d"]) * 100
                anomalies.append(Anomaly(
                    anomaly_id  = new_uuid(),
                    type        = "COST_SPIKE",
                    provider_id = row["provider_id"],
                    scenario_id = row["scenario_id"],
                    severity    = "WARNING",
                    details     = {
                        "cost_24h":  round(row["c_24h"], 6),
                        "cost_7d":   round(row["c_7d"],  6),
                        "increase_pct": round(pct, 1),
                        "threshold_pct": COST_SPIKE_THRESHOLD * 100,
                    },
                    created_at  = datetime.now(timezone.utc).isoformat(),
                ))
        return anomalies

    async def _check_safety_violations(self) -> list[Anomaly]:
        """Detect any eval_trace with safety_score < SAFETY_FLOOR in last 24h."""
        anomalies: list[Anomaly] = []
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT trace_id, scenario_id, safety_score,
                       (payload->>'provider_id') AS provider_id
                FROM eval_traces
                WHERE safety_score < $1
                  AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY safety_score ASC
                LIMIT 20
                """,
                SAFETY_FLOOR,
            )
            for row in rows:
                anomalies.append(Anomaly(
                    anomaly_id  = new_uuid(),
                    type        = "SAFETY_VIOLATION",
                    provider_id = row["provider_id"],
                    scenario_id = row["scenario_id"],
                    severity    = "CRITICAL",
                    details     = {
                        "trace_id":     row["trace_id"],
                        "safety_score": row["safety_score"],
                        "floor":        SAFETY_FLOOR,
                    },
                    created_at  = datetime.now(timezone.utc).isoformat(),
                ))
        return anomalies

    async def _check_thrashing(self) -> list[Anomaly]:
        """Detect routing thrashing: > THRASH_MAX_CHANGES provider changes in 24h."""
        anomalies: list[Anomaly] = []
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT COUNT(*) AS change_count
                FROM routing_decisions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                  AND reason LIKE '%escalation%'
                """
            )
            if rows and rows[0]["change_count"] > THRASH_MAX_CHANGES:
                count = rows[0]["change_count"]
                anomalies.append(Anomaly(
                    anomaly_id  = new_uuid(),
                    type        = "THRASHING",
                    provider_id = None,
                    scenario_id = None,
                    severity    = "WARNING",
                    details     = {
                        "provider_changes_24h": count,
                        "threshold":            THRASH_MAX_CHANGES,
                        "note":                 "Excessive routing instability detected.",
                    },
                    created_at  = datetime.now(timezone.utc).isoformat(),
                ))
        return anomalies

    async def _persist(self, anomaly: Anomaly) -> None:
        import json
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO anomaly_log
                        (anomaly_id, type, provider_id, scenario_id, severity, details)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    ON CONFLICT (anomaly_id) DO NOTHING
                    """,
                    anomaly.anomaly_id,
                    anomaly.type,
                    anomaly.provider_id,
                    anomaly.scenario_id,
                    anomaly.severity,
                    json.dumps(anomaly.details),
                )
        except Exception as exc:
            logger.debug("Anomaly persist skipped: %s", exc)
