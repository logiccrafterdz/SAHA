"""
SAHA – Privacy Gate (§7, Phase 2 M4)
Pre-flight check applied to every UnifiedAgentRequest before reaching any provider.

Responsibilities:
  1. PII Detection  — scan request text for PII patterns (regex + category tags)
  2. Redaction      — replace detected PII with type placeholders before forwarding
  3. Data Residency — validate provider is allowed for the task's residency requirements
  4. Output Scan    — scan provider response for leaked PII that slipped through redaction

All checks are logged to the audit trail. Blocking decisions emit a POLICY_ERROR.

Spec ref: §7 (Privacy & Data Governance)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from saha.contracts.common import CanonicalError, ErrorCode, ErrorSeverity, ErrorType, new_uuid

logger = logging.getLogger(__name__)

# ─── PII Patterns ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PIIPattern:
    name:        str
    pattern:     re.Pattern[str]
    placeholder: str
    severity:    str = "HIGH"  # HIGH | MEDIUM | LOW


_PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="EMAIL",
        pattern=re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
        placeholder="[REDACTED_EMAIL]",
        severity="HIGH",
    ),
    PIIPattern(
        name="PHONE_US",
        pattern=re.compile(r"\b(?:\+1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        placeholder="[REDACTED_PHONE]",
        severity="HIGH",
    ),
    PIIPattern(
        name="SSN",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        placeholder="[REDACTED_SSN]",
        severity="CRITICAL",
    ),
    PIIPattern(
        name="CREDIT_CARD",
        pattern=re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        placeholder="[REDACTED_CC]",
        severity="CRITICAL",
    ),
    PIIPattern(
        name="IP_ADDRESS",
        pattern=re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        placeholder="[REDACTED_IP]",
        severity="MEDIUM",
    ),
    PIIPattern(
        name="API_KEY",
        pattern=re.compile(
            r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AIza[A-Za-z0-9\-_]{35})\b"
        ),
        placeholder="[REDACTED_API_KEY]",
        severity="CRITICAL",
    ),
]


# ─── Data Residency Rules ─────────────────────────────────────────────────────

# Maps data_residency_tag → allowed provider_ids
_RESIDENCY_ALLOW_MAP: dict[str, set[str]] = {
    "us":     {"claude_3_5_sonnet", "gpt_4o", "gemini_1_5_pro"},
    "eu":     {"claude_3_5_sonnet", "gpt_4o"},
    "global": {"claude_3_5_sonnet", "gpt_4o", "gemini_1_5_pro"},
}


# ─── Detection Result ─────────────────────────────────────────────────────────

@dataclass
class PIIDetection:
    found:     bool
    types:     list[str]    = field(default_factory=list)
    severities: list[str]   = field(default_factory=list)
    redacted_text: str      = ""


# ─── Privacy Gate ─────────────────────────────────────────────────────────────

class PrivacyGate:
    """
    Stateless gate applied before every provider call and after every response.
    Inject into VendorGateway to enable for all requests.

    Usage:
        gate = PrivacyGate(redact=True)
        clean_request, detection = gate.check_request(request, provider_id, residency)
        if detection.found and not redact:
            return POLICY_ERROR
        # forward clean_request to provider
        response = await adapter.complete(clean_request)
        output_detection = gate.check_output(response.normalized_output.get("text",""))
    """

    def __init__(self, redact: bool = True) -> None:
        """
        redact=True:  replace PII with placeholders (default, non-blocking).
        redact=False: block request entirely if PII found (strict mode).
        """
        self._redact = redact

    # ── Request checks ────────────────────────────────────────────────────────

    def check_request(
        self,
        message:     str,
        system:      str,
        provider_id: str,
        residency:   list[str] | None = None,
    ) -> tuple[dict[str, str], PIIDetection | None]:
        """
        Scan request content for PII and validate data residency.
        Returns (cleaned_texts, PIIDetection | None).
        cleaned_texts = {'message': ..., 'system': ...}
        PIIDetection is None if no PII found.
        """
        # Residency check first (fast fail)
        if residency:
            violation = self._check_residency(provider_id, residency)
            if violation:
                logger.warning(
                    "PrivacyGate residency violation | provider=%s residency=%s",
                    provider_id, residency,
                )
                return (
                    {"message": message, "system": system},
                    PIIDetection(
                        found=True,
                        types=["RESIDENCY_VIOLATION"],
                        severities=["CRITICAL"],
                        redacted_text=message,
                    ),
                )

        # PII scan
        all_types:      list[str] = []
        all_severities: list[str] = []

        clean_message, types_m, sev_m = self._scan_and_redact(message)
        clean_system,  types_s, sev_s = self._scan_and_redact(system)

        all_types      = types_m + types_s
        all_severities = sev_m   + sev_s

        if all_types:
            detection = PIIDetection(
                found         = True,
                types         = list(dict.fromkeys(all_types)),
                severities    = list(dict.fromkeys(all_severities)),
                redacted_text = clean_message,
            )
            action = "redacted" if self._redact else "blocked"
            logger.warning(
                "PrivacyGate PII detected | provider=%s types=%s action=%s",
                provider_id, all_types, action,
            )
            return (
                {
                    "message": clean_message if self._redact else message,
                    "system":  clean_system  if self._redact else system,
                },
                detection,
            )

        return ({"message": message, "system": system}, None)

    def check_output(self, text: str) -> PIIDetection | None:
        """
        Scan provider output for leaked PII (post-generation guard).
        Returns PIIDetection if found, None if clean.
        """
        _, types, severities = self._scan_and_redact(text)
        if types:
            logger.warning("PrivacyGate output PII leak detected | types=%s", types)
            return PIIDetection(
                found      = True,
                types      = list(dict.fromkeys(types)),
                severities = list(dict.fromkeys(severities)),
            )
        return None

    def build_policy_error(self, detection: PIIDetection, provider_id: str) -> CanonicalError:
        """Build a CanonicalError for a privacy violation (blocking mode)."""
        return CanonicalError(
            type     = ErrorType.POLICY_ERROR,
            code     = ErrorCode.SAFETY_POLICY_VIOLATION,
            severity = ErrorSeverity.CRITICAL,
            details  = (
                f"PrivacyGate blocked request to provider='{provider_id}': "
                f"PII types detected = {detection.types}. "
                f"Redact or remove sensitive data before retrying."
            ),
        )


    # ── Internal helpers ──────────────────────────────────────────────────────

    def _scan_and_redact(
        self, text: str
    ) -> tuple[str, list[str], list[str]]:
        """
        Scan text for all PII patterns.
        Returns (redacted_text, detected_types, detected_severities).
        """
        if not text:
            return text, [], []

        found_types:      list[str] = []
        found_severities: list[str] = []
        redacted = text

        for pii in _PII_PATTERNS:
            matches = pii.pattern.findall(redacted)
            if matches:
                found_types.append(pii.name)
                found_severities.append(pii.severity)
                redacted = pii.pattern.sub(pii.placeholder, redacted)

        return redacted, found_types, found_severities

    def _check_residency(
        self,
        provider_id: str,
        residency:   list[str],
    ) -> bool:
        """Returns True if there's a residency violation (provider not allowed)."""
        for region in residency:
            allowed = _RESIDENCY_ALLOW_MAP.get(region, set())
            if provider_id not in allowed:
                return True
        return False

    # ── Convenience audit log ─────────────────────────────────────────────────

    def audit_log(
        self,
        task_id:     str,
        provider_id: str,
        detection:   PIIDetection | None,
        direction:   str = "request",  # 'request' | 'response'
    ) -> dict[str, Any]:
        """Build a structured audit record for the privacy event."""
        return {
            "audit_id":   new_uuid(),
            "task_id":    task_id,
            "provider_id": provider_id,
            "direction":  direction,
            "pii_found":  detection.found if detection else False,
            "pii_types":  detection.types if detection else [],
        }
