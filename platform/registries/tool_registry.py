"""ToolRegistry — maps tool_name to IToolAdapter instances."""

from __future__ import annotations

from platform.core.exceptions import ToolNotFound
from platform.core.interfaces.tool import IToolAdapter


class ToolRegistry:
    """In-memory registry mapping tool names to adapter instances.

    Populated at startup by ConfigLoader. Queried by AgentRuntime when
    resolving tool_use blocks returned by the LLM.
    """

    def __init__(self) -> None:
        self._store: dict[str, IToolAdapter] = {}

    def register(self, tool_name: str, adapter: IToolAdapter) -> None:
        """Register a tool adapter under the given name."""
        # TODO: implement
        raise NotImplementedError

    def get(self, tool_name: str) -> IToolAdapter:
        """Return IToolAdapter for the given name. Raises ToolNotFound if missing."""
        # TODO: implement
        raise NotImplementedError

    def list_all(self) -> list[str]:
        """Return all registered tool names."""
        # TODO: implement
        raise NotImplementedError
