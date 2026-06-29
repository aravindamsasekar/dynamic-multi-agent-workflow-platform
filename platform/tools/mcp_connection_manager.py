"""MCPConnectionManager — persistent stdio MCP server session with tool discovery."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:  # pragma: no cover — mcp optional at import time
    ClientSession = None  # type: ignore[assignment,misc]
    StdioServerParameters = None  # type: ignore[assignment,misc]
    stdio_client = None  # type: ignore[assignment]


class MCPToolNotFoundError(Exception):
    """Raised when a configured tool name does not exist on the MCP server."""


@dataclass
class MCPToolInfo:
    """Metadata for a tool discovered via list_tools()."""

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPConnectionManager:
    """Manages a persistent stdio MCP server process and reusable ClientSession.

    Lifecycle:
    - Connection is lazy: the subprocess starts on the first call_tool() call.
    - list_tools() is called immediately after session.initialize(); results are
      cached in _tools and used to validate tool names on every call.
    - The session is reused for all subsequent call_tool() calls.
    - On transport failure, the session is closed and reconnected once automatically.
    - close() shuts down the session and subprocess cleanly (called at app shutdown).
    """

    def __init__(self, server_command: str, server_args: list[str]) -> None:
        self._server_command = server_command
        self._server_args = server_args
        self._session: Any | None = None  # ClientSession, typed loosely for optional import
        self._exit_stack: AsyncExitStack | None = None
        self._tools: dict[str, MCPToolInfo] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call, reconnecting once on transport failure.

        Raises MCPToolNotFoundError if the tool is not in the discovered cache.
        Raises RuntimeError if the mcp package is not installed.
        """
        for attempt in range(2):
            try:
                session = await self._ensure_connected()
                if tool_name not in self._tools:
                    available = sorted(self._tools)
                    raise MCPToolNotFoundError(
                        f"Tool '{tool_name}' not found on MCP server. "
                        f"Available: {available}"
                    )
                return await session.call_tool(tool_name, arguments=arguments)
            except MCPToolNotFoundError:
                raise  # config error — never retry
            except Exception:
                if attempt == 1:
                    raise
                # Likely a transport/session error — disconnect and retry once
                await self._disconnect()

    @property
    def discovered_tools(self) -> dict[str, MCPToolInfo]:
        """Snapshot of tools discovered in the last list_tools() call."""
        return dict(self._tools)

    def has_tool(self, tool_name: str) -> bool:
        """Return True if the tool exists in the discovered cache."""
        return tool_name in self._tools

    async def close(self) -> None:
        """Shut down the session and subprocess cleanly."""
        await self._disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> Any:
        async with self._lock:
            if self._session is None:
                await self._connect()
            return self._session

    async def _connect(self) -> None:
        """Spawn the subprocess, initialize ClientSession, and discover tools."""
        if stdio_client is None:
            raise RuntimeError(
                "The 'mcp' package is required for MCP tool support. "
                "Install it with: pip install 'mcp>=1.0.0'"
            )
        stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self._server_command, args=self._server_args
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.list_tools()
            self._tools = {
                t.name: MCPToolInfo(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {},
                )
                for t in result.tools
            }
            self._session = session
            self._exit_stack = stack
        except Exception:
            await stack.aclose()
            raise

    async def _disconnect(self) -> None:
        async with self._lock:
            if self._exit_stack is not None:
                try:
                    await self._exit_stack.aclose()
                except Exception:
                    pass
                self._exit_stack = None
            self._session = None
            self._tools = {}
