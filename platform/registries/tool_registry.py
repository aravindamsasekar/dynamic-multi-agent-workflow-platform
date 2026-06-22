"""ToolRegistry — maps tool_name to IToolAdapter instances."""

from __future__ import annotations

from platform.core.exceptions import ToolNotFound
from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolDefinition


class ToolRegistry:
    """In-memory registry mapping tool names to adapter instances.

    Populated at startup by ConfigLoader. Queried by AgentRuntime when
    resolving tool_use blocks returned by the LLM.
    """

    def __init__(self) -> None:
        self._store: dict[str, IToolAdapter] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(
        self,
        tool_name: str,
        adapter: IToolAdapter,
        tool_def: ToolDefinition | None = None,
    ) -> None:
        """Register a tool adapter, and optionally its ToolDefinition for LLM tool prompting."""
        self._store[tool_name] = adapter
        if tool_def is not None:
            self._definitions[tool_name] = tool_def

    def get(self, tool_name: str) -> IToolAdapter:
        """Return IToolAdapter for the given name. Raises ToolNotFound if missing."""
        if tool_name not in self._store:
            raise ToolNotFound(f"Tool '{tool_name}' not found")
        return self._store[tool_name]

    def get_definition(self, tool_name: str) -> ToolDefinition | None:
        """Return ToolDefinition for the given name, or None if not registered."""
        return self._definitions.get(tool_name)

    def list_all(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._store.keys())

    def exists(self, tool_name: str) -> bool:
        """Return True if a tool with the given name is registered."""
        return tool_name in self._store

    def clear(self) -> None:
        """Remove all registered tool adapters and definitions."""
        self._store.clear()
        self._definitions.clear()
