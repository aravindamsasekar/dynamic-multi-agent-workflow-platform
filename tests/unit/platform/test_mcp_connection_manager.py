"""Unit tests for MCPConnectionManager."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform.tools.mcp_connection_manager import (
    MCPConnectionManager,
    MCPToolInfo,
    MCPToolNotFoundError,
)

_MOD = "platform.tools.mcp_connection_manager"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str = "", schema: dict | None = None) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object"}
    return t


def _make_call_result(text: str = "result text", is_error: bool = False) -> MagicMock:
    content_item = MagicMock()
    content_item.text = text
    r = MagicMock()
    r.content = [content_item]
    r.isError = is_error
    return r


def _make_session(
    tools: list | None = None,
    call_result: MagicMock | None = None,
) -> MagicMock:
    list_result = MagicMock()
    list_result.tools = tools if tools is not None else [_make_tool("read_file")]
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=list_result)
    session.call_tool = AsyncMock(return_value=call_result or _make_call_result())
    return session


def _make_stdio() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@contextmanager
def _mcp_patches(session: MagicMock, stdio: MagicMock):
    """Patch all three mcp module-level names needed for connection tests."""
    with patch(f"{_MOD}.StdioServerParameters"), \
         patch(f"{_MOD}.ClientSession", return_value=session), \
         patch(f"{_MOD}.stdio_client", return_value=stdio):
        yield


def _manager() -> MCPConnectionManager:
    return MCPConnectionManager(server_command="npx", server_args=["-y", "server"])


# ---------------------------------------------------------------------------
# TestMCPConnectionManagerInit
# ---------------------------------------------------------------------------


class TestMCPConnectionManagerInit:
    def test_stores_server_command(self) -> None:
        m = MCPConnectionManager(server_command="npx", server_args=[])
        assert m._server_command == "npx"

    def test_stores_server_args(self) -> None:
        m = MCPConnectionManager(server_command="node", server_args=["server.js", "--port", "0"])
        assert m._server_args == ["server.js", "--port", "0"]

    def test_empty_server_args(self) -> None:
        m = MCPConnectionManager(server_command="npx", server_args=[])
        assert m._server_args == []

    def test_session_none_initially(self) -> None:
        assert _manager()._session is None

    def test_tools_empty_initially(self) -> None:
        assert _manager()._tools == {}

    def test_has_tool_false_before_connect(self) -> None:
        assert _manager().has_tool("read_file") is False

    def test_discovered_tools_empty_before_connect(self) -> None:
        assert _manager().discovered_tools == {}


# ---------------------------------------------------------------------------
# TestMCPConnectionManagerConnect
# ---------------------------------------------------------------------------


class TestMCPConnectionManagerConnect:
    async def test_initialize_called_on_first_call(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
        session.initialize.assert_called_once()

    async def test_list_tools_called_on_connect(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
        session.list_tools.assert_called_once()

    async def test_tools_cached_after_connect(self) -> None:
        tools = [_make_tool("read_file"), _make_tool("write_file")]
        session = _make_session(tools=tools)
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
        assert "read_file" in m.discovered_tools
        assert "write_file" in m.discovered_tools

    async def test_tool_info_fields_populated(self) -> None:
        tool = _make_tool("read_file", "Read a file", {"type": "object", "properties": {"path": {}}})
        session = _make_session(tools=[tool])
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
        info = m.discovered_tools["read_file"]
        assert isinstance(info, MCPToolInfo)
        assert info.name == "read_file"
        assert info.description == "Read a file"
        assert "properties" in info.input_schema

    async def test_session_reused_across_calls(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
            await m.call_tool("read_file", {})
        # initialize and list_tools only once — session was reused
        session.initialize.assert_called_once()
        session.list_tools.assert_called_once()
        assert session.call_tool.call_count == 2

    async def test_has_tool_true_after_connect(self) -> None:
        session = _make_session(tools=[_make_tool("read_file")])
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
        assert m.has_tool("read_file") is True
        assert m.has_tool("nonexistent") is False

    async def test_discovered_tools_returns_snapshot(self) -> None:
        session = _make_session(tools=[_make_tool("read_file")])
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
        snapshot = m.discovered_tools
        snapshot["injected"] = MCPToolInfo("injected", "", {})
        assert "injected" not in m.discovered_tools  # original not mutated

    async def test_connect_failure_propagates(self) -> None:
        session = _make_session()
        session.initialize = AsyncMock(side_effect=RuntimeError("init failed"))
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            with pytest.raises(RuntimeError, match="init failed"):
                await m.call_tool("read_file", {})

    async def test_no_mcp_package_raises_runtime_error(self) -> None:
        m = _manager()
        with patch(f"{_MOD}.stdio_client", None):
            with pytest.raises(RuntimeError, match="mcp"):
                await m.call_tool("read_file", {})


# ---------------------------------------------------------------------------
# TestMCPConnectionManagerCallTool
# ---------------------------------------------------------------------------


class TestMCPConnectionManagerCallTool:
    async def test_call_tool_returns_mcp_result(self) -> None:
        result = _make_call_result("file contents")
        session = _make_session(call_result=result)
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            r = await m.call_tool("read_file", {"path": "README.md"})
        assert r is result

    async def test_arguments_forwarded_to_session(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {"path": "README.md", "encoding": "utf-8"})
        session.call_tool.assert_called_once_with(
            "read_file", arguments={"path": "README.md", "encoding": "utf-8"}
        )

    async def test_unknown_tool_raises_not_found(self) -> None:
        session = _make_session(tools=[_make_tool("read_file")])
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            with pytest.raises(MCPToolNotFoundError, match="write_file"):
                await m.call_tool("write_file", {})

    async def test_not_found_error_includes_available_tools(self) -> None:
        tools = [_make_tool("read_file"), _make_tool("list_dir")]
        session = _make_session(tools=tools)
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            with pytest.raises(MCPToolNotFoundError, match="read_file"):
                await m.call_tool("bad_tool", {})

    async def test_not_found_not_retried(self) -> None:
        session = _make_session(tools=[_make_tool("read_file")])
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            with pytest.raises(MCPToolNotFoundError):
                await m.call_tool("bad_tool", {})
        # initialize called only once — no reconnect attempted for config errors
        session.initialize.assert_called_once()

    async def test_reconnects_on_transport_error(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        call_count = 0

        async def _failing_then_succeeding(tool_name, *, arguments):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("broken pipe")
            return _make_call_result("recovered")

        session.call_tool = _failing_then_succeeding

        with _mcp_patches(session, stdio):
            m = _manager()
            result = await m.call_tool("read_file", {})
        assert result.content[0].text == "recovered"
        assert call_count == 2

    async def test_raises_after_two_transport_failures(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        session.call_tool = AsyncMock(side_effect=ConnectionError("broken"))

        with _mcp_patches(session, stdio):
            m = _manager()
            with pytest.raises(ConnectionError, match="broken"):
                await m.call_tool("read_file", {})


# ---------------------------------------------------------------------------
# TestMCPConnectionManagerClose
# ---------------------------------------------------------------------------


class TestMCPConnectionManagerClose:
    async def test_close_when_not_connected_no_error(self) -> None:
        await _manager().close()  # must not raise

    async def test_close_clears_session(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
            assert m._session is not None
            await m.close()
        assert m._session is None

    async def test_close_clears_tools(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
            await m.close()
        assert m._tools == {}

    async def test_close_calls_exit_stack(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
            await m.close()
        # stdio context manager __aexit__ was called (via AsyncExitStack.aclose)
        stdio.__aexit__.assert_called()

    async def test_double_close_no_error(self) -> None:
        session = _make_session()
        stdio = _make_stdio()
        with _mcp_patches(session, stdio):
            m = _manager()
            await m.call_tool("read_file", {})
            await m.close()
            await m.close()  # second close must not raise
