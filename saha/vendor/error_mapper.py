"""
SAHA – ErrorMapper: maps raw provider errors → canonical SAHA taxonomy.
Spec ref: §2.5
"""
from __future__ import annotations

import logging

import httpx

from saha.contracts.common import CanonicalError, ErrorCode, ErrorSeverity, ErrorType

logger = logging.getLogger(__name__)


class ErrorMapper:
    """
    Converts raw exceptions and HTTP status codes into canonical CanonicalError.
    All adapters call this; nothing leaks provider-specific error types upstream.
    """

    @staticmethod
    def from_exception(exc: Exception, provider_id: str = "") -> CanonicalError:
        details_prefix = f"[{provider_id}] " if provider_id else ""

        # ── HTTP / network errors ────────────────────────────────────────────
        if isinstance(exc, httpx.TimeoutException):
            return CanonicalError(
                type=ErrorType.INFRA_ERROR,
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                severity=ErrorSeverity.WARNING,
                details=f"{details_prefix}Request timed out: {exc}",
            )

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return ErrorMapper.from_http_status(status, details=str(exc), provider_id=provider_id)

        # ── Anthropic-specific ───────────────────────────────────────────────
        exc_name = type(exc).__name__
        if "RateLimitError" in exc_name:
            return CanonicalError(
                type=ErrorType.INFRA_ERROR,
                code=ErrorCode.PROVIDER_RATE_LIMIT,
                severity=ErrorSeverity.WARNING,
                details=f"{details_prefix}{exc}",
            )

        if "AuthenticationError" in exc_name:
            return CanonicalError(
                type=ErrorType.POLICY_ERROR,
                code=ErrorCode.SAFETY_POLICY_VIOLATION,
                severity=ErrorSeverity.CRITICAL,
                details=f"{details_prefix}Authentication failed: {exc}",
            )

        if "BadRequestError" in exc_name:
            return CanonicalError(
                type=ErrorType.MODEL_ERROR,
                code=ErrorCode.REFUSAL_INFO,
                severity=ErrorSeverity.INFO,
                details=f"{details_prefix}Bad request: {exc}",
            )

        # ── Fallback: unknown ────────────────────────────────────────────────
        logger.warning("Unknown error from %s: %s – %s", provider_id, exc_name, exc)
        return CanonicalError(
            type=ErrorType.INFRA_ERROR,
            code=ErrorCode.UNKNOWN,
            severity=ErrorSeverity.WARNING,
            details=f"{details_prefix}Unclassified error ({exc_name}): {exc}",
        )

    @staticmethod
    def from_http_status(
        status: int,
        details: str = "",
        provider_id: str = "",
    ) -> CanonicalError:
        prefix = f"[{provider_id}] " if provider_id else ""
        full_details = f"{prefix}HTTP {status}: {details}"

        if status == 429:
            return CanonicalError(
                type=ErrorType.INFRA_ERROR,
                code=ErrorCode.PROVIDER_RATE_LIMIT,
                severity=ErrorSeverity.WARNING,
                details=full_details,
            )
        if status in (401, 403):
            return CanonicalError(
                type=ErrorType.POLICY_ERROR,
                code=ErrorCode.SAFETY_POLICY_VIOLATION,
                severity=ErrorSeverity.CRITICAL,
                details=full_details,
            )
        if status >= 500:
            return CanonicalError(
                type=ErrorType.INFRA_ERROR,
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                severity=ErrorSeverity.WARNING,
                details=full_details,
            )

        return CanonicalError(
            type=ErrorType.INFRA_ERROR,
            code=ErrorCode.UNKNOWN,
            severity=ErrorSeverity.WARNING,
            details=full_details,
        )

    @staticmethod
    def budget_exceeded(budget_cap: float, budget_used: float) -> CanonicalError:
        return CanonicalError(
            type=ErrorType.POLICY_ERROR,
            code=ErrorCode.BUDGET_EXCEEDED,
            severity=ErrorSeverity.CRITICAL,
            details=f"Job stopped due to budget cap {budget_cap:.2f} (used {budget_used:.4f})",
        )
