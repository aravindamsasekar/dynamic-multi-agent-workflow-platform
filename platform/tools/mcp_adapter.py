"""MCPAdapter — thin IToolAdapter that delegates to MCPConnectionManager."""

from __future__ import annotations

from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolCall, ToolResult
from platform.tools.mcp_connection_manager import (
    MCPConnectionManager,
    MCPToolNotFoundError,
)


class MCPAdapter(IToolAdapter):
    """Sends tool calls to an MCP server via a managed stdio connection.

    Configured via adapter_config in tools.yaml:
        server_command: npx
        server_args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
        tool_name: read_file

    All connection lifecycle is managed by MCPConnectionManager:
    the subprocess starts on first use, the session is reused across calls,
    and the connection is closed cleanly at application shutdown.
    """

    def __init__(
        self,
        server_command: str,
        tool_name: str,
        server_args: list[str] | None = None,
        *,
        _manager: MCPConnectionManager | None = None,
    ) -> None:
        self._tool_name = tool_name
        self._manager = _manager or MCPConnectionManager(
            server_command=server_command,
            server_args=server_args or [],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            result = await self._manager.call_tool(self._tool_name, call.input)
        except MCPToolNotFoundError as exc:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=str(exc),
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=f"MCP tool call failed: {exc}",
                is_error=True,
            )

        text_parts = [p.text for p in result.content if hasattr(p, "text")]
        return ToolResult(
            tool_use_id=call.tool_use_id,
            content="\n".join(text_parts),
            is_error=bool(result.isError),
        )

    async def close(self) -> None:
        await self._manager.close()
