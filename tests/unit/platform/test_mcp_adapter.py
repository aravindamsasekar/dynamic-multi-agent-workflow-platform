"""Unit tests for MCPAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from platform.core.models.tool import ToolCall, ToolResult
from platform.tools.mcp_adapter import MCPAdapter
from platform.tools.mcp_connection_manager import MCPToolNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(tool_use_id: str = "tu-1", tool_name: str = "filesystem_read_file", **inputs) -> ToolCall:
    return ToolCall(tool_use_id=tool_use_id, tool_name=tool_name, input=inputs)


def _make_content(*texts: str) -> list[MagicMock]:
    items = []
    for t in texts:
        item = MagicMock()
        item.text = t
        items.append(item)
    return items


def _make_mcp_result(texts: list[str] | None = None, is_error: bool = False) -> MagicMock:
    r = MagicMock()
    r.content = _make_content(*(texts or ["result"]))
    r.isError = is_error
    return r


def _make_manager(result=None, side_effect=None) -> MagicMock:
    m = MagicMock()
    if side_effect is not None:
        m.call_tool = AsyncMock(side_effect=side_effect)
    else:
        m.call_tool = AsyncMock(return_value=result or _make_mcp_result())
    m.close = AsyncMock()
    return m


def _adapter(tool_name: str = "read_file", manager=None, **kwargs) -> MCPAdapter:
    return MCPAdapter(
        server_command="npx",
        tool_name=tool_name,
        server_args=["-y", "server"],
        _manager=manager or _make_manager(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# TestMCPAdapterInit
# ---------------------------------------------------------------------------


class TestMCPAdapterInit:
    def test_stores_tool_name(self) -> None:
        adapter = _adapter(tool_name="read_file")
        assert adapter._tool_name == "read_file"

    def test_uses_injected_manager(self) -> None:
        manager = _make_manager()
        adapter = _adapter(manager=manager)
        assert adapter._manager is manager

    def test_creates_manager_when_not_injected(self) -> None:
        from platform.tools.mcp_connection_manager import MCPConnectionManager
        adapter = MCPAdapter(server_command="npx", tool_name="read_file", server_args=["-y", "x"])
        assert isinstance(adapter._manager, MCPConnectionManager)

    def test_server_args_default_empty(self) -> None:
        from platform.tools.mcp_connection_manager import MCPConnectionManager
        adapter = MCPAdapter(server_command="npx", tool_name="read_file")
        assert adapter._manager._server_args == []


# ---------------------------------------------------------------------------
# TestMCPAdapterExecute
# ---------------------------------------------------------------------------


class TestMCPAdapterExecute:
    async def test_success_returns_text_content(self) -> None:
        manager = _make_manager(result=_make_mcp_result(["file contents"]))
        adapter = _adapter(tool_name="read_file", manager=manager)
        result = await adapter.execute(_make_call(path="README.md"))
        assert result.content == "file contents"
        assert result.is_error is False

    async def test_multiple_text_parts_joined(self) -> None:
        manager = _make_manager(result=_make_mcp_result(["part one", "part two"]))
        adapter = _adapter(manager=manager)
        result = await adapter.execute(_make_call())
        assert result.content == "part one\npart two"

    async def test_empty_content_list_returns_empty_string(self) -> None:
        mcp_result = MagicMock()
        mcp_result.content = []
        mcp_result.isError = False
        adapter = _adapter(manager=_make_manager(result=mcp_result))
        result = await adapter.execute(_make_call())
        assert result.content == ""
        assert result.is_error is False

    async def test_non_text_content_skipped(self) -> None:
        text_item = MagicMock(spec=["text"])
        text_item.text = "hello"
        image_item = MagicMock(spec=[])  # no .text attribute
        mcp_result = MagicMock()
        mcp_result.content = [image_item, text_item]
        mcp_result.isError = False
        adapter = _adapter(manager=_make_manager(result=mcp_result))
        result = await adapter.execute(_make_call())
        assert result.content == "hello"

    async def test_is_error_propagated_from_mcp(self) -> None:
        manager = _make_manager(result=_make_mcp_result(["tool error msg"], is_error=True))
        adapter = _adapter(manager=manager)
        result = await adapter.execute(_make_call())
        assert result.is_error is True
        assert result.content == "tool error msg"

    async def test_tool_use_id_propagated_on_success(self) -> None:
        adapter = _adapter(manager=_make_manager())
        result = await adapter.execute(_make_call(tool_use_id="tu-xyz"))
        assert result.tool_use_id == "tu-xyz"

    async def test_tool_use_id_propagated_on_not_found(self) -> None:
        adapter = _adapter(manager=_make_manager(side_effect=MCPToolNotFoundError("no such tool")))
        result = await adapter.execute(_make_call(tool_use_id="tu-err"))
        assert result.tool_use_id == "tu-err"
        assert result.is_error is True

    async def test_tool_use_id_propagated_on_transport_exception(self) -> None:
        adapter = _adapter(manager=_make_manager(side_effect=ConnectionError("broken")))
        result = await adapter.execute(_make_call(tool_use_id="tu-crash"))
        assert result.tool_use_id == "tu-crash"
        assert result.is_error is True

    async def test_not_found_returns_error_result(self) -> None:
        adapter = _adapter(
            tool_name="bad_tool",
            manager=_make_manager(side_effect=MCPToolNotFoundError("Tool 'bad_tool' not found")),
        )
        result = await adapter.execute(_make_call())
        assert result.is_error is True
        assert "bad_tool" in result.content

    async def test_transport_exception_returns_error_result(self) -> None:
        adapter = _adapter(manager=_make_manager(side_effect=RuntimeError("server crashed")))
        result = await adapter.execute(_make_call())
        assert result.is_error is True
        assert "server crashed" in result.content

    async def test_input_forwarded_to_manager(self) -> None:
        manager = _make_manager()
        adapter = _adapter(tool_name="read_file", manager=manager)
        await adapter.execute(_make_call(path="README.md", encoding="utf-8"))
        manager.call_tool.assert_called_once_with(
            "read_file", {"path": "README.md", "encoding": "utf-8"}
        )

    async def test_tool_name_forwarded_to_manager(self) -> None:
        manager = _make_manager()
        adapter = _adapter(tool_name="list_dir", manager=manager)
        await adapter.execute(_make_call())
        name_arg = manager.call_tool.call_args[0][0]
        assert name_arg == "list_dir"


# ---------------------------------------------------------------------------
# TestMCPAdapterClose
# ---------------------------------------------------------------------------


class TestMCPAdapterClose:
    async def test_close_delegates_to_manager(self) -> None:
        manager = _make_manager()
        adapter = _adapter(manager=manager)
        await adapter.close()
        manager.close.assert_called_once()

    async def test_close_is_idempotent(self) -> None:
        manager = _make_manager()
        adapter = _adapter(manager=manager)
        await adapter.close()
        await adapter.close()
        assert manager.close.call_count == 2
