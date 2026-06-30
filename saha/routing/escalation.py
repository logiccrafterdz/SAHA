"""
SAHA – Escalation Policy (§4.4)
Detects when a provider should be replaced mid-task or on next task.

Rules (§4.4):
  1. quality_score < threshold for N consecutive tasks in a scenario → escalate
  2. error_type in CRITICAL_ERRORS → escalate immediately
  3. budget exhausted with cheaper provider available → escalate
  4. Explicit HITL override → escalate

Cooldown (§4.4.3):
  After switching, the new provider must serve M tasks before another switch.
  Prevents oscillation/thrashing.

Spec ref: §4.4 (Stability & Escalation Policies)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from saha.contracts.common import new_uuid
from saha.contracts.routing import EscalationEvent, EscalationTrigger
from saha.db.connection import get_pool

logger = logging.getLogger(__name__)


@dataclass
class _ProviderTracker:
    """In-memory tracker per (provider_id, scenario_id) pair."""
    consecutive_failures:   int = 0
    tasks_since_switch:     int = 0
    last_provider:          str = ""
    cooldown_remaining:     int = 0


class EscalationPolicy:
    """
    Tracks provider performance per scenario and decides when to escalate.
    Maintains in-memory state; persists escalation events to DB.

    Usage:
        policy = EscalationPolicy()
        # After each eval result:
        should, event = await policy.check(provider_id, scenario_id, quality, safety, error_type)
        if should:
            next_provider = pick_from_fallback(event)
    """

    def __init__(self, trigger: EscalationTrigger | None = None) -> None:
        self._trigger  = trigger or EscalationTrigger()
        # keyed by (provider_id, scenario_id)
        self._trackers: dict[tuple[str, str], _ProviderTracker] = defaultdict(
            _ProviderTracker
        )

    async def check(
        self,
        task_id:     str,
        provider_id: str,
        scenario_id: str,
        quality:     int,
        safety:      int,
        error_type:  str = "NONE",
        cost_exceeded: bool = False,
        fallback_provider_id: str | None = None,
    ) -> tuple[bool, EscalationEvent | None]:
        """
        Evaluate whether escalation is needed after a task result.
        Returns (should_escalate, EscalationEvent | None).
        The caller decides which provider to use next based on EscalationEvent.
        """
        key     = (provider_id, scenario_id)
        tracker = self._trackers[key]
        tracker.tasks_since_switch += 1

        # ── Cooldown guard ───────────────────────────────────────────────────
        if tracker.cooldown_remaining > 0:
            tracker.cooldown_remaining -= 1
            logger.debug(
                "Escalation cooldown active for %s/%s: %d tasks remaining",
                provider_id, scenario_id, tracker.cooldown_remaining,
            )
            return False, None

        # ── Check 1: Critical error → immediate escalation ───────────────────
        for critical in self._trigger.critical_error_triggers:
            if error_type.startswith(critical.split(".")[0]) and "CRITICAL" in error_type:
                event = await self._escalate(
                    task_id, provider_id, fallback_provider_id,
                    trigger="SAFETY_VIOLATION" if "SAFETY" in error_type else "CRITICAL_ERROR",
                    reason=f"Critical error '{error_type}' triggered immediate escalation.",
                    severity="CRITICAL",
                )
                self._reset_tracker(key)
                return True, event

        # ── Check 2: Safety score below floor ───────────────────────────────
        if safety < self._trigger.safety_min:
            event = await self._escalate(
                task_id, provider_id, fallback_provider_id,
                trigger="SAFETY_VIOLATION",
                reason=(
                    f"Safety score {safety} < threshold {self._trigger.safety_min} "
                    f"for {provider_id}/{scenario_id}."
                ),
                severity="CRITICAL",
            )
            self._reset_tracker(key)
            return True, event

        # ── Check 3: Quality below threshold → track consecutive ─────────────
        if quality < self._trigger.quality_threshold:
            tracker.consecutive_failures += 1
            logger.debug(
                "Quality failure %d/%d for %s/%s (score=%d)",
                tracker.consecutive_failures,
                self._trigger.consecutive_failures,
                provider_id, scenario_id, quality,
            )
            if tracker.consecutive_failures >= self._trigger.consecutive_failures:
                event = await self._escalate(
                    task_id, provider_id, fallback_provider_id,
                    trigger="CONSECUTIVE_FAILURES",
                    reason=(
                        f"{tracker.consecutive_failures} consecutive quality failures "
                        f"(threshold={self._trigger.quality_threshold}) for "
                        f"{provider_id}/{scenario_id}."
                    ),
                    severity="WARNING",
                )
                self._reset_tracker(key)
                return True, event
        else:
            # Quality OK → reset consecutive failure counter
            tracker.consecutive_failures = 0

        # ── Check 4: Budget exhausted ────────────────────────────────────────
        if cost_exceeded and fallback_provider_id:
            event = await self._escalate(
                task_id, provider_id, fallback_provider_id,
                trigger="BUDGET",
                reason=(
                    f"Budget cap exceeded for {provider_id}. "
                    f"Escalating to cheaper provider '{fallback_provider_id}'."
                ),
                severity="WARNING",
            )
            self._reset_tracker(key)
            return True, event

        return False, None

    def record_provider_switch(
        self,
        from_provider: str,
        to_provider:   str,
        scenario_id:   str,
    ) -> None:
        """
        Record that a provider switch happened (from escalation).
        Activates cooldown on the new provider to prevent thrashing (§4.4.3).
        """
        new_key = (to_provider, scenario_id)
        tracker = self._trackers[new_key]
        tracker.cooldown_remaining  = self._trigger.cooldown_tasks
        tracker.last_provider       = from_provider
        tracker.tasks_since_switch  = 0
        logger.info(
            "Provider switch recorded: %s → %s | cooldown=%d tasks",
            from_provider, to_provider, self._trigger.cooldown_tasks,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    async def _escalate(
        self,
        task_id:      str,
        from_provider: str,
        to_provider:   str | None,
        trigger:      str,
        reason:       str,
        severity:     str = "WARNING",
    ) -> EscalationEvent:
        event = EscalationEvent(
            task_id          = task_id,
            from_provider_id = from_provider,
            to_provider_id   = to_provider or "",
            trigger          = trigger,
            reason           = reason,
            severity         = severity,
        )
        logger.warning(
            "Escalation triggered | trigger=%s from=%s to=%s reason=%s",
            trigger, from_provider, to_provider, reason,
        )
        await self._persist_event(event)
        return event

    def _reset_tracker(self, key: tuple[str, str]) -> None:
        tracker = self._trackers[key]
        tracker.consecutive_failures = 0

    async def _persist_event(self, event: EscalationEvent) -> None:
        """Log escalation to routing_decisions with reason='escalation'."""
        payload = json.dumps({
            "trigger":       event.trigger,
            "from_provider": event.from_provider_id,
            "to_provider":   event.to_provider_id,
            "severity":      event.severity,
        })
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO routing_decisions
                        (decision_id, task_id, chosen_provider_id,
                         fallback_provider_id, mode, reason, payload)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb)
                    """,
                    event.event_id,
                    event.task_id,
                    event.to_provider_id or event.from_provider_id,
                    event.from_provider_id,
                    "escalation",
                    event.reason,
                    payload,
                )
        except Exception as exc:
            logger.debug("Escalation DB persist skipped: %s", exc)
