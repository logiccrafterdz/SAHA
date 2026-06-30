"""
SAHA – Eval Trace Storage (Hot/Warm/Cold tiered).
Persists EvalTrace to PostgreSQL.  Privacy Gate stubs included.
Spec ref: §1.5, §5.3
"""
from __future__ import annotations

import json
import logging
from typing import Any

from saha.contracts.eval import EvalInput, EvalResult, EvalTrace, StorageTier
from saha.contracts.common import new_uuid
from saha.db.connection import get_pool

logger = logging.getLogger(__name__)

# PII token replacements (Privacy Gate – §5.3)
_PII_PATTERNS: list[tuple[str, str]] = [
    # (regex_pattern, replacement)
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    (r"\b\d{9,}\b", "[ID_NUMBER]"),
]


class EvalTraceStorage:
    """
    Writes EvalTrace records to the eval_traces table (PostgreSQL).
    Applies lightweight Privacy Gate scrubbing before persistence.
    Spec ref: §1.5 (Tiered Storage), §5.3 (Privacy Gate)
    """

    async def save(
        self,
        eval_input: EvalInput,
        eval_result: EvalResult,
        raw_output: dict[str, Any] | None = None,
        allow_training: bool = False,
    ) -> EvalTrace:
        trace = EvalTrace(
            trace_id=new_uuid(),
            eval_input=eval_input,
            eval_result=eval_result,
            raw_output=raw_output or {},
            storage_tier=StorageTier.HOT,
            allow_training=allow_training,
        )

        # Apply Privacy Gate (passive transformation, never drops)
        scrubbed_input  = self._scrub(eval_input.to_bus_payload())
        scrubbed_result = eval_result.to_bus_payload()
        scrubbed_raw    = self._scrub(raw_output or {})

        pool = await get_pool()
        provider_id = (
            eval_input.provider_info.provider_id
            if eval_input.provider_info else None
        )
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO eval_traces
                    (trace_id, eval_id, scenario_id, provider_id,
                     task_type, final_verdict, quality_score, safety_score,
                     latency_ms, cost_incurred, storage_tier, allow_training,
                     eval_input, eval_result, raw_output)
                VALUES
                    ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,
                     $13::jsonb, $14::jsonb, $15::jsonb)
                """,
                trace.trace_id,
                eval_input.eval_id,
                eval_input.scenario_id,
                provider_id,
                eval_input.task_type,
                eval_result.final_verdict,
                eval_result.quality_score,
                eval_result.safety_score,
                eval_result.latency_ms,
                eval_result.cost_incurred,
                StorageTier.HOT,
                allow_training,
                json.dumps(scrubbed_input),
                json.dumps(scrubbed_result),
                json.dumps(scrubbed_raw),
            )

        logger.info(
            "EvalTrace saved | trace_id=%s verdict=%s quality=%d",
            trace.trace_id, eval_result.final_verdict, eval_result.quality_score,
        )
        return trace

    async def promote_to_warm(self, trace_id: str) -> None:
        """
        HOT → WARM: drop raw_output, keep normalized data + metrics.
        Spec ref: §1.5 (Warm Storage)
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE eval_traces
                SET storage_tier = 'WARM',
                    raw_output   = '{}'::jsonb,
                    promoted_at  = NOW()
                WHERE trace_id = $1 AND storage_tier = 'HOT'
                """,
                trace_id,
            )
        logger.info("EvalTrace %s promoted HOT → WARM", trace_id)

    async def promote_to_cold(self, trace_id: str) -> None:
        """
        WARM → COLD: PII-scrub eval_input, keep only task_type / scenario /
        verdict / metrics.  Spec ref: §1.5 (Cold/Training Storage)
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE eval_traces
                SET storage_tier = 'COLD',
                    eval_input   = jsonb_build_object(
                        'task_type',   eval_input->>'task_type',
                        'scenario_id', eval_input->>'scenario_id',
                        'domain_tags', eval_input->'domain_tags'
                    ),
                    promoted_at  = NOW()
                WHERE trace_id = $1 AND storage_tier = 'WARM'
                """,
                trace_id,
            )
        logger.info("EvalTrace %s promoted WARM → COLD (PII scrubbed)", trace_id)

    # ── Privacy Gate helpers (§5.3) ──────────────────────────────────────────

    def _scrub(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Lightweight PII scrubbing on dict serialised to JSON string.
        Passive: transforms data, never drops it.
        """
        import re
        text = json.dumps(data)
        for pattern, replacement in _PII_PATTERNS:
            text = re.sub(pattern, replacement, text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return data  # fallback: return original if scrubbing corrupted JSON
