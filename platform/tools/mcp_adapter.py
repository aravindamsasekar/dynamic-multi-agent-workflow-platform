"""MCPAdapter — executes tool calls via the Model Context Protocol."""

from __future__ import annotations

from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolCall, ToolResult


class MCPAdapter(IToolAdapter):
    """Sends tool calls to an MCP server using the MCP protocol.

    Configured via adapter_config in tools.yaml:
        server_url: "http://localhost:8001"
    """

    def __init__(self, server_url: str) -> None:
        self._server_url = server_url
        # TODO Phase 4: initialize MCP client using the mcp SDK

    async def execute(self, call: ToolCall) -> ToolResult:
        # TODO Phase 4: implement MCP protocol call
        raise NotImplementedError("MCPAdapter is not implemented in Phase 3")
