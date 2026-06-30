"""
SAHA – Vendor Abstraction Layer: registry and dispatch  [Phase 2]
Upper layers call VendorGateway.complete(); they never import adapters directly.
Registered providers: claude_3_5_sonnet | gpt_4o | gemini_1_5_pro
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderProfile,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)
from saha.vendor.adapters.claude import ClaudeAdapter
from saha.vendor.base import BaseAdapter

try:
    from saha.vendor.adapters.openai import OpenAIAdapter as _OpenAIAdapter
    _HAVE_OPENAI = True
except ImportError:
    _HAVE_OPENAI = False

try:
    from saha.vendor.adapters.gemini import GeminiAdapter as _GeminiAdapter
    _HAVE_GEMINI = True
except ImportError:
    _HAVE_GEMINI = False

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VendorGateway:
    """
    Registry of all available adapters.
    Dispatches requests to the correct adapter based on provider_id.
    Phase 2: auto-registers GPT-4o and Gemini if their SDKs are installed.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}
        # Always register Claude (Phase 1 baseline)
        self._register(ClaudeAdapter())
        # Register Phase 2 adapters if SDKs are present
        if _HAVE_OPENAI:
            try:
                self._register(_OpenAIAdapter())
            except Exception as exc:
                logger.warning("GPT-4o adapter skipped: %s", exc)
        if _HAVE_GEMINI:
            try:
                self._register(_GeminiAdapter())
            except Exception as exc:
                logger.warning("Gemini adapter skipped: %s", exc)

    def _register(self, adapter: BaseAdapter) -> None:
        self._adapters[adapter.provider_id] = adapter
        logger.info("Registered adapter: %s", adapter.provider_id)

    def available_providers(self) -> list[str]:
        return list(self._adapters.keys())

    def get_profile(self, provider_id: str) -> ProviderProfile:
        adapter = self._get_adapter(provider_id)
        return adapter.get_profile()

    async def complete(
        self,
        provider_id: str,
        request: UnifiedAgentRequest,
    ) -> UnifiedAgentResponse:
        """Route request to the named provider adapter."""
        adapter = self._get_adapter(provider_id)
        logger.info(
            "Dispatching request_id=%s to provider=%s",
            request.request_id,
            provider_id,
        )
        return await adapter.complete(request)

    async def interrupt(
        self,
        provider_id: str,
        signal: BudgetInterruptSignal,
    ) -> bool:
        adapter = self._get_adapter(provider_id)
        return await adapter.interrupt(signal)

    def _get_adapter(self, provider_id: str) -> BaseAdapter:
        if provider_id not in self._adapters:
            raise ValueError(
                f"No adapter registered for provider_id='{provider_id}'. "
                f"Available: {list(self._adapters)}"
            )
        return self._adapters[provider_id]


# Module-level singleton
_gateway: VendorGateway | None = None


def get_gateway() -> VendorGateway:
    global _gateway
    if _gateway is None:
        _gateway = VendorGateway()
    return _gateway
