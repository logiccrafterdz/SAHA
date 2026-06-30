"""
SAHA – Claude (Anthropic) adapter.
Translates UnifiedAgentRequest ↔ Anthropic SDK types and maps responses
into UnifiedAgentResponse. Only this file knows about the Anthropic SDK.
Spec ref: §2.1–2.5
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

import anthropic
from pydantic_settings import BaseSettings, SettingsConfigDict

from saha.contracts.common import CanonicalError, ErrorCode, ErrorType, ErrorSeverity, new_uuid
from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderCapabilities,
    ProviderPolicies,
    ProviderPricing,
    ProviderProfile,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)
from saha.vendor.base import BaseAdapter
from saha.vendor.error_mapper import ErrorMapper

logger = logging.getLogger(__name__)

_PROFILE_PATH = pathlib.Path(__file__).parent.parent / "profiles" / "claude.json"

# Claude model IDs we support
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS_DEFAULT = 4096


class ClaudeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ANTHROPIC_API_KEY: str = ""


class ClaudeAdapter(BaseAdapter):
    """
    Anthropic Claude adapter for SAHA Vendor Abstraction Layer.
    Supports single-turn and multi-turn (via short_term memory) completions,
    tool-use detection, and budget interrupt.
    """

    def __init__(self) -> None:
        settings = ClaudeSettings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._profile = self._load_profile()

    # ── BaseAdapter interface ────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return "claude_3_5_sonnet"

    def get_profile(self) -> ProviderProfile:
        return self._profile

    async def complete(self, request: UnifiedAgentRequest) -> UnifiedAgentResponse:
        """Send request to Claude; return normalised SAHA response."""
        run_id = new_uuid()
        start_ms = int(time.monotonic() * 1000)

        # Build messages from memory + current message
        messages = self._build_messages(request)
        tools    = self._build_tools(request)

        try:
            response = await self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS_DEFAULT,
                system=request.system_prompt or anthropic.NOT_GIVEN,
                messages=messages,
                tools=tools if tools else anthropic.NOT_GIVEN,
            )
        except Exception as exc:
            error = ErrorMapper.from_exception(exc, provider_id=self.provider_id)
            return UnifiedAgentResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                run_id=run_id,
                status="FAILED",
                error=error,
                latency_ms=int(time.monotonic() * 1000) - start_ms,
            )

        latency_ms = int(time.monotonic() * 1000) - start_ms
        return self._parse_response(response, request.request_id, run_id, latency_ms)

    async def interrupt(self, signal: BudgetInterruptSignal) -> bool:
        """
        Anthropic doesn't expose a cancel API for ongoing requests.
        Budget interrupt is handled at the Execution Harness level
        by not starting a new step if the budget is exhausted.
        We return False (not cancellable mid-flight) and log it.
        """
        logger.warning(
            "Budget interrupt requested for run_id=%s (provider=claude). "
            "Claude does not support mid-flight cancellation; "
            "next step will be blocked at Execution Harness level.",
            signal.run_id,
        )
        return False

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_messages(
        self, request: UnifiedAgentRequest
    ) -> list[dict[str, Any]]:
        """
        Combine short-term memory (previous turns) with the current message.
        Memory entries are assumed to already be in Anthropic message format.
        """
        history: list[dict[str, Any]] = []

        # Re-hydrate previous conversation turns from short_term memory
        # Each memory entry: {"role": "user"|"assistant", "content": "..."}
        from saha.contracts.execution import AgentMemory  # avoid circular at module level

        # If the request carries agent_state_id, short_term is available via AgentState.
        # For simplicity in the adapter, we accept raw dicts injected by the Execution Harness.
        # The Harness injects extra_context into options as a convention (Phase 1).
        extra_ctx: list[dict[str, Any]] = []  # placeholder; Harness sets this via options

        history = extra_ctx or []
        history.append({"role": "user", "content": request.message})
        return history

    def _build_tools(
        self, request: UnifiedAgentRequest
    ) -> list[dict[str, Any]]:
        """Convert SAHA ToolSchema list into Anthropic tool format."""
        if not request.tools:
            return []
        return [
            {
                "name": t.name,
                "description": t.input_schema.get("description", ""),
                "input_schema": t.input_schema.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in request.tools
        ]

    def _parse_response(
        self,
        response: anthropic.types.Message,
        request_id: str,
        run_id: str,
        latency_ms: int,
    ) -> UnifiedAgentResponse:
        """Translate Anthropic Message → UnifiedAgentResponse."""
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Cost estimate ($ per token)
        cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

        # Detect tool-use blocks
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks     = [b for b in response.content if b.type == "text"]

        normalized_output: dict[str, Any] = {
            "text": " ".join(b.text for b in text_blocks) if text_blocks else "",
            "stop_reason": response.stop_reason,
        }

        if tool_use_blocks:
            # First tool call (SAHA processes one at a time)
            tool_block = tool_use_blocks[0]
            pending_tool: dict[str, Any] = {
                "name": tool_block.name,
                "arguments": tool_block.input,
                "tool_use_id": tool_block.id,
            }
            return UnifiedAgentResponse(
                request_id=request_id,
                provider_id=self.provider_id,
                run_id=run_id,
                status="NEEDS_TOOL",
                normalized_output=normalized_output,
                tool_calls_count=len(tool_use_blocks),
                context_tokens_used=input_tokens + output_tokens,
                cost_estimate=cost,
                latency_ms=latency_ms,
                pending_tool_call=pending_tool,
                error=CanonicalError.none(),
            )

        # Normal completion
        return UnifiedAgentResponse(
            request_id=request_id,
            provider_id=self.provider_id,
            run_id=run_id,
            status="COMPLETED",
            normalized_output=normalized_output,
            tool_calls_count=0,
            context_tokens_used=input_tokens + output_tokens,
            cost_estimate=cost,
            latency_ms=latency_ms,
            error=CanonicalError.none(),
        )

    def _load_profile(self) -> ProviderProfile:
        data = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
        return ProviderProfile(**data)
