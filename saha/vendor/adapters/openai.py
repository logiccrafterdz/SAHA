"""
SAHA – GPT-4o (OpenAI) adapter.
Translates UnifiedAgentRequest ↔ OpenAI SDK types.
Only this file knows about the OpenAI SDK.
Spec ref: §2.1–2.5

Pricing (gpt-4o, as of 2024):
  Input:  $2.50 / 1M tokens
  Output: $10.00 / 1M tokens

Tool-use mapping:
  SAHA ToolSchema.name + input_schema → OpenAI function tool format.
  Response: choices[0].message.tool_calls → NEEDS_TOOL status.
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

_PROFILE_PATH  = pathlib.Path(__file__).parent.parent / "profiles" / "openai.json"

GPT4O_MODEL        = "gpt-4o"
MAX_TOKENS_DEFAULT = 4096

# Cost per token ($/token)
_INPUT_COST_PER_TOKEN  = 2.50  / 1_000_000
_OUTPUT_COST_PER_TOKEN = 10.00 / 1_000_000


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    OPENAI_API_KEY: str = ""


class OpenAIAdapter(BaseAdapter):
    """
    OpenAI GPT-4o adapter for SAHA Vendor Abstraction Layer.
    Lazy-imports openai so the SDK is optional (not required for Claude-only deployments).
    """

    def __init__(self) -> None:
        settings = OpenAISettings()
        self._api_key = settings.OPENAI_API_KEY
        self._client  = None    # lazy init — avoids import at module load
        self._profile = self._load_profile()

    # ── BaseAdapter interface ────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return "gpt_4o"

    def get_profile(self) -> ProviderProfile:
        return self._profile

    async def complete(self, request: UnifiedAgentRequest) -> UnifiedAgentResponse:
        """Send request to GPT-4o; return normalised SAHA response."""
        run_id   = new_uuid()
        start_ms = int(time.monotonic() * 1000)

        client = self._get_client()
        messages = self._build_messages(request)
        tools    = self._build_tools(request)

        try:
            kwargs: dict[str, Any] = {
                "model":      GPT4O_MODEL,
                "max_tokens": MAX_TOKENS_DEFAULT,
                "messages":   messages,
            }
            if tools:
                kwargs["tools"]        = tools
                kwargs["tool_choice"]  = "auto"
            if request.system_prompt:
                # OpenAI expects system as first message
                messages.insert(0, {"role": "system", "content": request.system_prompt})

            response = await client.chat.completions.create(**kwargs)

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
        """OpenAI does not support mid-flight cancellation via API."""
        logger.warning(
            "Budget interrupt requested for run_id=%s (provider=gpt_4o). "
            "OpenAI does not support mid-flight cancellation.",
            signal.run_id,
        )
        return False

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for GPT-4o adapter. "
                    "Install with: pip install openai"
                ) from exc
        return self._client

    def _build_messages(self, request: UnifiedAgentRequest) -> list[dict[str, Any]]:
        """Build OpenAI messages array from request."""
        return [{"role": "user", "content": request.message}]

    def _build_tools(self, request: UnifiedAgentRequest) -> list[dict[str, Any]]:
        """Convert SAHA ToolSchema list into OpenAI function tool format."""
        return self._build_tools_from_saha(request.tools)

    def _build_tools_from_saha(self, tools: list) -> list[dict[str, Any]]:
        """Testable helper: convert ToolSchema list → OpenAI function format."""
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name":        t.name,
                    "description": t.input_schema.get("description", ""),
                    "parameters":  t.input_schema.get("input_schema", {
                        "type": "object", "properties": {}
                    }),
                },
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
        """Translate OpenAI ChatCompletion → UnifiedAgentResponse."""
        choice  = response.choices[0]
        message = choice.message
        usage   = response.usage

        input_tokens  = usage.prompt_tokens     if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = input_tokens * _INPUT_COST_PER_TOKEN + output_tokens * _OUTPUT_COST_PER_TOKEN

        normalized_output: dict[str, Any] = {
            "text":        message.content or "",
            "stop_reason": choice.finish_reason,
        }

        # Tool call detection
        if message.tool_calls:
            tc = message.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}

            pending_tool: dict[str, Any] = {
                "name":        tc.function.name,
                "arguments":   args,
                "tool_use_id": tc.id,
            }
            return UnifiedAgentResponse(
                request_id       = request_id,
                provider_id      = self.provider_id,
                run_id           = run_id,
                status           = "NEEDS_TOOL",
                normalized_output= normalized_output,
                tool_calls_count = len(message.tool_calls),
                context_tokens_used = input_tokens + output_tokens,
                cost_estimate    = cost,
                latency_ms       = latency_ms,
                pending_tool_call= pending_tool,
                error            = CanonicalError.none(),
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
