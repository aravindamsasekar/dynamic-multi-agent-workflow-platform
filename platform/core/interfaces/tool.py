"""IToolAdapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from platform.core.models.tool import ToolCall, ToolResult


class IToolAdapter(ABC):
    """Abstract interface for tool execution adapters."""

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        ...

    async def close(self) -> None:
        """Release any resources held by this adapter. No-op by default."""
