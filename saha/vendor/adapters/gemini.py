"""
SAHA – Gemini 1.5 Pro (Google) adapter.
Translates UnifiedAgentRequest ↔ google-generativeai SDK types.
Only this file knows about the Google SDK.
Spec ref: §2.1–2.5

Pricing (gemini-1.5-pro, as of 2024):
  Input:  $1.25 / 1M tokens  (prompts ≤ 128K)
  Output: $5.00 / 1M tokens

Tool-use mapping:
  SAHA ToolSchema → Google FunctionDeclaration format.
  Response: candidate.content.parts with function_call → NEEDS_TOOL status.
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from saha.contracts.common import CanonicalError, new_uuid
from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderProfile,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)
from saha.vendor.base import BaseAdapter
from saha.vendor.error_mapper import ErrorMapper

logger = logging.getLogger(__name__)

_PROFILE_PATH = pathlib.Path(__file__).parent.parent / "profiles" / "gemini.json"

GEMINI_MODEL       = "gemini-1.5-pro"
MAX_TOKENS_DEFAULT = 4096

_INPUT_COST_PER_TOKEN  = 1.25 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 5.00 / 1_000_000


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    GOOGLE_API_KEY: str = ""


class GeminiAdapter(BaseAdapter):
    """
    Google Gemini 1.5 Pro adapter for SAHA Vendor Abstraction Layer.
    Lazy-imports google-generativeai so the SDK is optional.
    """

    def __init__(self) -> None:
        settings      = GeminiSettings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model   = None   # lazy init
        self._profile = self._load_profile()

    # ── BaseAdapter interface ────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return "gemini_1_5_pro"

    def get_profile(self) -> ProviderProfile:
        return self._profile

    async def complete(self, request: UnifiedAgentRequest) -> UnifiedAgentResponse:
        """Send request to Gemini 1.5 Pro; return normalised SAHA response."""
        run_id   = new_uuid()
        start_ms = int(time.monotonic() * 1000)

        model   = self._get_model(request.tools)
        prompt  = self._build_prompt(request)

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": MAX_TOKENS_DEFAULT,
                    "temperature":       0.0,
                },
            )
        except Exception as exc:
            error = ErrorMapper.from_exception(exc, provider_id=self.provider_id)
            return UnifiedAgentResponse(
                request_id  = request.request_id,
                provider_id = self.provider_id,
                run_id      = run_id,
                status      = "FAILED",
                error       = error,
                latency_ms  = int(time.monotonic() * 1000) - start_ms,
            )

        latency_ms = int(time.monotonic() * 1000) - start_ms
        return self._parse_response(response, request.request_id, run_id, latency_ms)

    async def interrupt(self, signal: BudgetInterruptSignal) -> bool:
        logger.warning(
            "Budget interrupt for run_id=%s (provider=gemini_1_5_pro). "
            "Gemini does not support mid-flight cancellation.",
            signal.run_id,
        )
        return False

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_model(self, tools: list | None = None) -> Any:
        """Lazy-init the Gemini GenerativeModel with optional tool config."""
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package required for Gemini adapter. "
                "Install with: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=self._api_key)

        tool_config = None
        if tools:
            declarations = self._build_tools(tools)
            if declarations:
                tool_config = declarations

        return genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            tools=tool_config,
        )

    def _build_prompt(self, request: UnifiedAgentRequest) -> list[dict[str, Any]]:
        """Build Gemini content list (system + user turn)."""
        parts = []
        if request.system_prompt:
            parts.append({"role": "user",  "parts": [request.system_prompt]})
            parts.append({"role": "model", "parts": ["Understood."]})
        parts.append({"role": "user", "parts": [request.message]})
        return parts

    def _build_tools(self, tools: list) -> list[dict[str, Any]]:
        """Convert SAHA ToolSchema → Google FunctionDeclaration format."""
        return [
            {
                "function_declarations": [
                    {
                        "name":        t.name,
                        "description": t.input_schema.get("description", ""),
                        "parameters":  t.input_schema.get("input_schema", {
                            "type": "object", "properties": {}
                        }),
                    }
                ]
            }
            for t in tools
        ]

    def _parse_response(
        self,
        response:   Any,
        request_id: str,
        run_id:     str,
        latency_ms: int,
    ) -> UnifiedAgentResponse:
        """Translate Gemini GenerateContentResponse → UnifiedAgentResponse."""
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None:
            return UnifiedAgentResponse(
                request_id  = request_id,
                provider_id = self.provider_id,
                run_id      = run_id,
                status      = "FAILED",
                error       = ErrorMapper.from_exception(
                    ValueError("Gemini returned empty candidates"),
                    provider_id=self.provider_id,
                ),
                latency_ms  = latency_ms,
            )

        # Usage (Gemini returns usage_metadata on the response object)
        usage          = getattr(response, "usage_metadata", None)
        input_tokens   = getattr(usage, "prompt_token_count",     0) if usage else 0
        output_tokens  = getattr(usage, "candidates_token_count", 0) if usage else 0
        cost = input_tokens * _INPUT_COST_PER_TOKEN + output_tokens * _OUTPUT_COST_PER_TOKEN

        # Check for function call parts
        text_parts     = []
        fc_parts       = []
        for part in (candidate.content.parts if candidate.content else []):
            if hasattr(part, "function_call") and part.function_call:
                fc_parts.append(part)
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        normalized_output: dict[str, Any] = {
            "text":        " ".join(text_parts),
            "stop_reason": str(candidate.finish_reason) if candidate.finish_reason else "STOP",
        }

        if fc_parts:
            fc = fc_parts[0].function_call
            pending_tool: dict[str, Any] = {
                "name":        fc.name,
                "arguments":   dict(fc.args),
                "tool_use_id": new_uuid(),   # Gemini doesn't assign tool IDs
            }
            return UnifiedAgentResponse(
                request_id          = request_id,
                provider_id         = self.provider_id,
                run_id              = run_id,
                status              = "NEEDS_TOOL",
                normalized_output   = normalized_output,
                tool_calls_count    = len(fc_parts),
                context_tokens_used = input_tokens + output_tokens,
                cost_estimate       = cost,
                latency_ms          = latency_ms,
                pending_tool_call   = pending_tool,
                error               = CanonicalError.none(),
            )

        return UnifiedAgentResponse(
            request_id          = request_id,
            provider_id         = self.provider_id,
            run_id              = run_id,
            status              = "COMPLETED",
            normalized_output   = normalized_output,
            tool_calls_count    = 0,
            context_tokens_used = input_tokens + output_tokens,
            cost_estimate       = cost,
            latency_ms          = latency_ms,
            error               = CanonicalError.none(),
        )

    def _load_profile(self) -> ProviderProfile:
        data = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
        return ProviderProfile(**data)
