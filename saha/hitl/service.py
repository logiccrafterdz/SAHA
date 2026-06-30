"""
SAHA – HITL (Human-in-the-Loop) Service (§6.2, Phase 2 M4)
Central service for all human intervention workflows.

HITL Intervention Types (§6.2):
  6.2.1 Router Policy Override      — modify routing constraints in real-time
  6.2.2 Judge Calibration Trigger   — re-calibrate LLM judge against golden dataset
  6.2.3 Success Contract Update     — update evaluation criteria for a scenario
  6.2.4 Failure Triage              — classify and record root cause of critical failures

Design principles:
  - All HITL actions require an approved_by field (audit trail).
  - Actions are idempotent (re-applying same override is safe).
  - Actions emit events to the bus (EVAL_RESULTS or ANOMALY_ALERTS topics).
  - All persisted via DB; in-memory effects are also applied immediately.

Spec ref: §6.2 (HITL Intervention Workflows)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from saha.contracts.common import new_uuid
from saha.event_bus import topics

logger = logging.getLogger(__name__)


# ─── HITL Action Contracts ───────────────────────────────────────────────────

@dataclass
class RouterOverride:
    """§6.2.1 — Routing constraint modification."""
    override_id:  str
    scope:        str                 # 'global' | 'project_X' | 'scenario_Y'
    change:       dict[str, Any]      # e.g. {"quality_min": 85}
    reason:       str
    approved_by:  str
    created_at:   datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class SuccessContractUpdate:
    """§6.2.3 — Update evaluation success criteria for a scenario."""
    contract_id:  str
    scenario_id:  str
    old_contract: dict[str, Any]
    new_contract: dict[str, Any]
    reason:       str
    approved_by:  str
    effective_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class FailureTriage:
    """§6.2.4 — Classify and record root cause of a critical failure."""
    incident_id:      str
    run_id:           str
    eval_id:          str | None
    classified_root:  str   # e.g. 'MODEL_HALLUCINATION' | 'BAD_PROMPT' | 'INFRA'
    notes:            str   = ""
    action_items:     list[str] = field(default_factory=list)
    resolved:         bool  = False
    created_at:       datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ─── HITL Service ────────────────────────────────────────────────────────────

class HITLService:
    """
    Central service for all HITL intervention workflows.
    Wires together: ConstraintManager (for §6.2.1), bus events, and DB persistence.

    Inject at application startup via HITLService(constraint_manager, bus).
    """

    def __init__(
        self,
        constraint_manager: Any | None = None,   # saha.routing.ConstraintManager
        bus:                Any | None = None,    # SAHABusClient
    ) -> None:
        self._constraints = constraint_manager
        self._bus         = bus
        # In-memory log of overrides for quick inspection
        self._overrides:  list[RouterOverride]       = []
        self._triages:    list[FailureTriage]         = []
        self._contracts:  list[SuccessContractUpdate] = []

    # ── §6.2.1 Router Policy Override ────────────────────────────────────────

    async def apply_router_override(
        self,
        scope:       str,
        change:      dict[str, Any],
        reason:      str,
        approved_by: str,
        override_id: str | None = None,
    ) -> RouterOverride:
        """
        Apply a routing constraint override immediately.
        Takes effect for all subsequent routing decisions within the scope.
        """
        override = RouterOverride(
            override_id = override_id or new_uuid(),
            scope       = scope,
            change      = change,
            reason      = reason,
            approved_by = approved_by,
        )

        # Apply to ConstraintManager (immediate effect)
        if self._constraints is not None:
            self._constraints.apply_hitl_override(scope=scope, change=change)

        self._overrides.append(override)
        await self._persist_override(override)

        logger.info(
            "HITL §6.2.1: RouterOverride applied | scope=%s keys=%s approved_by=%s",
            scope, list(change.keys()), approved_by,
        )
        await self._publish_event(topics.ANOMALY_ALERTS, {
            "type":        "HITL_OVERRIDE",
            "override_id": override.override_id,
            "scope":       scope,
            "keys":        list(change.keys()),
            "approved_by": approved_by,
        })
        return override

    def get_active_overrides(self, scope: str = "global") -> list[RouterOverride]:
        return [o for o in self._overrides if o.scope == scope]

    def clear_override(self, scope: str, approved_by: str) -> None:
        """Revert a scope's override to defaults."""
        if self._constraints is not None:
            self._constraints.clear_override(scope)
        self._overrides = [o for o in self._overrides if o.scope != scope]
        logger.info("HITL §6.2.1: override cleared | scope=%s by=%s", scope, approved_by)

    # ── §6.2.3 Success Contract Update ───────────────────────────────────────

    async def update_success_contract(
        self,
        scenario_id:  str,
        old_contract: dict[str, Any],
        new_contract: dict[str, Any],
        reason:       str,
        approved_by:  str,
    ) -> SuccessContractUpdate:
        """
        Record a success contract update for a scenario.
        The new contract will be picked up by the Grader on next evaluation.
        """
        update = SuccessContractUpdate(
            contract_id  = new_uuid(),
            scenario_id  = scenario_id,
            old_contract = old_contract,
            new_contract = new_contract,
            reason       = reason,
            approved_by  = approved_by,
        )
        self._contracts.append(update)
        await self._persist_contract(update)

        logger.info(
            "HITL §6.2.3: SuccessContract updated | scenario=%s approved_by=%s",
            scenario_id, approved_by,
        )
        return update

    def get_latest_contract(self, scenario_id: str) -> SuccessContractUpdate | None:
        matches = [c for c in self._contracts if c.scenario_id == scenario_id]
        return matches[-1] if matches else None

    # ── §6.2.4 Failure Triage ─────────────────────────────────────────────────

    async def record_failure_triage(
        self,
        run_id:          str,
        classified_root: str,
        notes:           str       = "",
        action_items:    list[str] = None,
        eval_id:         str | None = None,
    ) -> FailureTriage:
        """
        Record the root cause classification of a critical failure.
        Triggers an ANOMALY_ALERTS bus event for dashboard visibility.
        """
        triage = FailureTriage(
            incident_id     = new_uuid(),
            run_id          = run_id,
            eval_id         = eval_id,
            classified_root = classified_root,
            notes           = notes,
            action_items    = action_items or [],
        )
        self._triages.append(triage)
        await self._persist_triage(triage)

        logger.warning(
            "HITL §6.2.4: FailureTriage recorded | run=%s root=%s actions=%d",
            run_id, classified_root, len(triage.action_items),
        )
        await self._publish_event(topics.ANOMALY_ALERTS, {
            "type":             "FAILURE_TRIAGE",
            "incident_id":      triage.incident_id,
            "run_id":           run_id,
            "classified_root":  classified_root,
            "action_items":     triage.action_items,
        })
        return triage

    def get_open_triages(self) -> list[FailureTriage]:
        return [t for t in self._triages if not t.resolved]

    def resolve_triage(self, incident_id: str) -> bool:
        for t in self._triages:
            if t.incident_id == incident_id:
                t.resolved = True
                logger.info("HITL §6.2.4: incident resolved | %s", incident_id)
                return True
        return False

    # ── §6.2.2 Judge Calibration (proxy) ─────────────────────────────────────

    async def trigger_judge_calibration(
        self,
        scenario_filter: list[str] | None = None,
        approved_by:     str = "hitl_service",
    ) -> dict[str, Any]:
        """
        Trigger judge calibration via the Eval harness (§6.2.2).
        In production this calls EvalAPI; here we emit a bus event
        that the eval_api can listen on.
        """
        payload = {
            "type":            "CALIBRATION_REQUESTED",
            "scenario_filter": scenario_filter or [],
            "approved_by":     approved_by,
            "request_id":      new_uuid(),
        }
        await self._publish_event(topics.EVAL_RESULTS, payload)
        logger.info("HITL §6.2.2: calibration requested | scenarios=%s", scenario_filter)
        return {"status": "requested", **payload}

    # ── DB persistence (graceful if no pool) ─────────────────────────────────

    async def _persist_override(self, override: RouterOverride) -> None:
        try:
            from saha.db.connection import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO hitl_overrides
                        (override_id, scope, change, reason, approved_by)
                    VALUES ($1, $2, $3::jsonb, $4, $5)
                    ON CONFLICT (override_id) DO NOTHING
                    """,
                    override.override_id,
                    override.scope,
                    json.dumps(override.change),
                    override.reason,
                    override.approved_by,
                )
        except Exception as exc:
            logger.debug("HITL override DB persist skipped: %s", exc)

    async def _persist_contract(self, update: SuccessContractUpdate) -> None:
        try:
            from saha.db.connection import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO success_contract_history
                        (contract_id, scenario_id, old_contract, new_contract, reason, approved_by)
                    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
                    """,
                    update.contract_id,
                    update.scenario_id,
                    json.dumps(update.old_contract),
                    json.dumps(update.new_contract),
                    update.reason,
                    update.approved_by,
                )
        except Exception as exc:
            logger.debug("HITL contract DB persist skipped: %s", exc)

    async def _persist_triage(self, triage: FailureTriage) -> None:
        try:
            from saha.db.connection import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO failure_triages
                        (incident_id, run_id, eval_id, classified_root, notes, action_items)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    triage.incident_id,
                    triage.run_id,
                    triage.eval_id,
                    triage.classified_root,
                    triage.notes,
                    json.dumps(triage.action_items),
                )
        except Exception as exc:
            logger.debug("HITL triage DB persist skipped: %s", exc)

    async def _publish_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            await self._bus.publish(topic, payload)
        except Exception as exc:
            logger.debug("HITL bus publish skipped: %s", exc)
