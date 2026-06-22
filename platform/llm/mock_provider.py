"""MockLLMProvider — deterministic provider for tests."""

from __future__ import annotations

from collections import deque

from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.message import LLMResponse, Message
from platform.core.models.tool import ToolDefinition


class MockLLMProvider(ILLMProvider):
    """Deterministic LLM provider that returns pre-loaded responses in order.

    Raises RuntimeError if complete() is called after the queue is exhausted.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._queue: deque[LLMResponse] = deque(responses)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        if not self._queue:
            raise RuntimeError("MockLLMProvider response queue is exhausted")
        return self._queue.popleft()
