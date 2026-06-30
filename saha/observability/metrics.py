"""
SAHA – Metrics Aggregator (§5.1, §5.2)
Reads raw traces from PostgreSQL and produces per-provider, per-window aggregates
stored in provider_stats. These aggregates are consumed by the Cost Router (§4.3).

Windows: '24h', '7d', '30d'
Metrics per (provider_id, scenario_id, window):
  quality_p50, quality_p90, safety_avg, success_rate, error_rate,
  cost_per_task, latency_p50_ms, latency_p90_ms, sample_count

Usage:
  aggregator = MetricsAggregator()
  await aggregator.compute_all()               # refresh all windows
  stats = await aggregator.get_provider_stats("claude_3_5_sonnet")
  report = await aggregator.get_cross_provider_report("SCENARIO_PY_FIX")

Spec ref: §4.3 (recent_eval_stats input to Cost Router), §5.1 (reports)
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from saha.contracts.common import new_uuid
from saha.db.connection import get_pool

logger = logging.getLogger(__name__)

WINDOWS = ["24h", "7d", "30d"]

_WINDOW_HOURS = {"24h": 24, "7d": 168, "30d": 720}


@dataclass
class ProviderWindowStats:
    provider_id:    str
    scenario_id:    str
    window:         str
    quality_p50:    float = 0.0
    quality_p90:    float = 0.0
    safety_avg:     float = 0.0
    success_rate:   float = 0.0
    error_rate:     float = 0.0
    cost_per_task:  float = 0.0
    latency_p50_ms: int   = 0
    latency_p90_ms: int   = 0
    sample_count:   int   = 0
    computed_at:    str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id":    self.provider_id,
            "scenario_id":    self.scenario_id,
            "window":         self.window,
            "quality_p50":    round(self.quality_p50, 1),
            "quality_p90":    round(self.quality_p90, 1),
            "safety_avg":     round(self.safety_avg,  1),
            "success_rate":   round(self.success_rate, 3),
            "error_rate":     round(self.error_rate,   3),
            "cost_per_task":  round(self.cost_per_task, 6),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "sample_count":   self.sample_count,
            "computed_at":    self.computed_at,
        }


class MetricsAggregator:
    """
    Computes and persists provider performance aggregates.
    Reads from eval_traces; writes to provider_stats.
    """

    async def compute_all(self) -> list[ProviderWindowStats]:
        """Refresh all (provider, scenario, window) combinations."""
        all_stats: list[ProviderWindowStats] = []
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                pairs = await conn.fetch(
                    """
                    SELECT DISTINCT
                        (payload->>'provider_id') AS provider_id,
                        scenario_id
                    FROM eval_traces
                    WHERE payload->>'provider_id' IS NOT NULL
                    """
                )
                for row in pairs:
                    provider_id = row["provider_id"]
                    scenario_id = row["scenario_id"]
                    for window in WINDOWS:
                        stats = await self._compute_window(
                            conn, provider_id, scenario_id, window
                        )
                        await self._upsert_stats(conn, stats)
                        all_stats.append(stats)
            logger.info("MetricsAggregator: computed %d stats entries", len(all_stats))
        except Exception as exc:
            logger.warning("MetricsAggregator skipped (no DB): %s", exc)
        return all_stats

    async def get_provider_stats(
        self,
        provider_id: str,
        window: str = "7d",
    ) -> list[dict[str, Any]]:
        """Get all scenario stats for a provider in the given window."""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM provider_stats
                    WHERE provider_id = $1 AND window = $2
                    ORDER BY scenario_id
                    """,
                    provider_id, window,
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("get_provider_stats skipped: %s", exc)
            return []

    async def get_all_providers_summary(
        self,
        window: str = "7d",
    ) -> list[dict[str, Any]]:
        """Get a cross-provider summary (all scenarios aggregated) for the window."""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        provider_id,
                        window,
                        AVG(quality_p50)    AS quality_p50,
                        AVG(quality_p90)    AS quality_p90,
                        AVG(safety_avg)     AS safety_avg,
                        AVG(success_rate)   AS success_rate,
                        AVG(cost_per_task)  AS cost_per_task,
                        AVG(latency_p50_ms) AS latency_p50_ms,
                        SUM(sample_count)   AS total_samples,
                        MAX(computed_at)    AS last_computed
                    FROM provider_stats
                    WHERE window = $1
                    GROUP BY provider_id, window
                    ORDER BY quality_p50 DESC
                    """,
                    window,
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("get_all_providers_summary skipped: %s", exc)
            return []

    async def get_cross_provider_report(
        self,
        scenario_id: str,
        window: str = "7d",
    ) -> dict[str, Any]:
        """
        Compare all providers on a specific scenario.
        Used by Cost Router to select optimal provider.
        """
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM provider_stats
                    WHERE scenario_id = $1 AND window = $2
                    ORDER BY quality_p50 DESC
                    """,
                    scenario_id, window,
                )
                providers = {r["provider_id"]: dict(r) for r in rows}
                return {
                    "scenario_id": scenario_id,
                    "window":      window,
                    "providers":   providers,
                }
        except Exception as exc:
            logger.debug("get_cross_provider_report skipped: %s", exc)
            return {"scenario_id": scenario_id, "window": window, "providers": {}}

    # ── Private ───────────────────────────────────────────────────────────────

    async def _compute_window(
        self,
        conn: Any,
        provider_id: str,
        scenario_id: str,
        window: str,
    ) -> ProviderWindowStats:
        hours = _WINDOW_HOURS[window]
        rows = await conn.fetch(
            """
            SELECT
                final_verdict,
                quality_score,
                safety_score,
                cost_incurred,
                latency_ms,
                error_type
            FROM eval_traces
            WHERE scenario_id = $1
              AND created_at > NOW() - ($2 || ' hours')::INTERVAL
              AND payload->>'provider_id' = $3
            """,
            scenario_id, str(hours), provider_id,
        )

        if not rows:
            return ProviderWindowStats(
                provider_id=provider_id,
                scenario_id=scenario_id,
                window=window,
            )

        qualities  = [r["quality_score"] for r in rows]
        safeties   = [r["safety_score"]  for r in rows]
        costs      = [r["cost_incurred"] for r in rows]
        latencies  = [r["latency_ms"]    for r in rows]
        verdicts   = [r["final_verdict"] for r in rows]
        n          = len(rows)

        return ProviderWindowStats(
            provider_id    = provider_id,
            scenario_id    = scenario_id,
            window         = window,
            quality_p50    = _percentile(qualities, 50),
            quality_p90    = _percentile(qualities, 90),
            safety_avg     = statistics.mean(safeties) if safeties else 0.0,
            success_rate   = verdicts.count("SUCCESS") / n,
            error_rate     = (verdicts.count("FAILURE") / n),
            cost_per_task  = statistics.mean(costs)     if costs else 0.0,
            latency_p50_ms = int(_percentile(latencies, 50)),
            latency_p90_ms = int(_percentile(latencies, 90)),
            sample_count   = n,
        )

    async def _upsert_stats(self, conn: Any, stats: ProviderWindowStats) -> None:
        await conn.execute(
            """
            INSERT INTO provider_stats
                (provider_id, scenario_id, window,
                 quality_p50, quality_p90, safety_avg,
                 success_rate, error_rate, cost_per_task,
                 latency_p50_ms, latency_p90_ms, sample_count, computed_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW())
            ON CONFLICT (provider_id, scenario_id, window)
            DO UPDATE SET
                quality_p50    = EXCLUDED.quality_p50,
                quality_p90    = EXCLUDED.quality_p90,
                safety_avg     = EXCLUDED.safety_avg,
                success_rate   = EXCLUDED.success_rate,
                error_rate     = EXCLUDED.error_rate,
                cost_per_task  = EXCLUDED.cost_per_task,
                latency_p50_ms = EXCLUDED.latency_p50_ms,
                latency_p90_ms = EXCLUDED.latency_p90_ms,
                sample_count   = EXCLUDED.sample_count,
                computed_at    = NOW()
            """,
            stats.provider_id,
            stats.scenario_id,
            stats.window,
            stats.quality_p50,
            stats.quality_p90,
            stats.safety_avg,
            stats.success_rate,
            stats.error_rate,
            stats.cost_per_task,
            stats.latency_p50_ms,
            stats.latency_p90_ms,
            stats.sample_count,
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _percentile(data: list[float | int], p: int) -> float:
    """Return the p-th percentile of data (0–100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])
