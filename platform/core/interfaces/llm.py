"""ILLMProvider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from platform.core.models.message import LLMResponse, Message
from platform.core.models.tool import ToolDefinition


class ILLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a structured response."""
        ...
