"""Claude LLM provider using the Anthropic SDK."""

from __future__ import annotations

from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.message import LLMResponse, Message
from platform.core.models.tool import ToolDefinition


class ClaudeProvider(ILLMProvider):
    """Anthropic Claude implementation of ILLMProvider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._model = model
        # TODO: initialize anthropic.AsyncAnthropic client

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        # TODO: implement
        raise NotImplementedError
