"""
SAHA – BaseAdapter: abstract base class for all provider adapters.
Spec ref: §2.1 (Vendor Abstraction responsibilities).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from saha.contracts.vendor import (
    BudgetInterruptSignal,
    ProviderProfile,
    UnifiedAgentRequest,
    UnifiedAgentResponse,
)


class BaseAdapter(ABC):
    """
    Every provider adapter must implement this interface.
    Upper layers (Execution Harness, Cost Router) only ever call these methods;
    they never import provider-specific code directly.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique stable identifier for this provider (e.g. 'claude_3_5_sonnet')."""
        ...

    @abstractmethod
    async def complete(self, request: UnifiedAgentRequest) -> UnifiedAgentResponse:
        """
        Send a request to the provider and return a normalised response.
        Must handle all provider-specific translation internally.
        Must never raise – always return a response (with error field set on failure).
        """
        ...

    @abstractmethod
    async def interrupt(self, signal: BudgetInterruptSignal) -> bool:
        """
        Attempt to stop an in-flight job due to budget exceeded.
        Returns True if successfully cancelled, False otherwise.
        """
        ...

    @abstractmethod
    def get_profile(self) -> ProviderProfile:
        """Return the static capability & policy profile for this provider."""
        ...
