"""
SAHA – Normalization Pipeline.
Strips provider-specific metadata and produces a canonical normalized_output.
Spec ref: §1.4
"""
from __future__ import annotations

import logging
import re
from typing import Any

from saha.contracts.common import CanonicalError, ErrorCode, ErrorSeverity, ErrorType

logger = logging.getLogger(__name__)

# Provider-specific keys to strip from raw output
_PROVIDER_META_KEYS = {
    "id", "model", "type", "role", "stop_reason", "stop_sequence",
    "usage", "x_request_id", "request_id", "system_fingerprint",
}


class NormalizationPipeline:
    """
    Converts raw provider output into a canonical normalized_output dict.
    Any failure here is recorded as EVAL_ERROR.NORMALIZATION_FAILED.CRITICAL (§1.4).
    """

    def normalize(
        self,
        raw_output: dict[str, Any],
        provider_id: str = "",
    ) -> tuple[dict[str, Any], CanonicalError | None]:
        """
        Returns (normalized_output, error).
        If error is not None, normalization failed and the eval should be aborted.
        """
        try:
            normalized = self._strip_metadata(raw_output)
            normalized = self._canonicalize(normalized)
            return normalized, None
        except Exception as exc:
            error = CanonicalError(
                type=ErrorType.EVAL_ERROR,
                code=ErrorCode.NORMALIZATION_FAILED,
                severity=ErrorSeverity.CRITICAL,
                details=f"[{provider_id}] Normalization failed: {exc}",
            )
            logger.error("Normalization failed for provider=%s: %s", provider_id, exc)
            return {}, error

    def _strip_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Remove provider-specific wrapper keys."""
        return {k: v for k, v in raw.items() if k not in _PROVIDER_META_KEYS}

    def _canonicalize(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Canonicalize structure:
        - Ensure 'text' key exists (extract from nested content if needed)
        - Sort list values for deterministic comparison
        - Convert None → ""
        """
        result: dict[str, Any] = {}

        for key, value in data.items():
            if value is None:
                result[key] = ""
            elif isinstance(value, list):
                # Flatten simple string lists; leave complex structures as-is
                if all(isinstance(v, str) for v in value):
                    result[key] = sorted(value)
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self._canonicalize(value)
            else:
                result[key] = value

        # Guarantee 'text' key exists
        if "text" not in result:
            # Try common aliases
            for alias in ("content", "message", "output", "answer", "response"):
                if alias in result:
                    result["text"] = str(result[alias])
                    break
            else:
                result["text"] = ""

        return result
