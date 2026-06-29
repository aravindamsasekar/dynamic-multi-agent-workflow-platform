"""Unit tests for MockAdapter, HTTPAdapter, and MCPAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from platform.core.models.tool import ToolCall
from platform.tools.http_adapter import HTTPAdapter
from platform.tools.mcp_adapter import MCPAdapter
from platform.tools.mock_adapter import MockAdapter


def make_call(tool_use_id: str = "tu-1", tool_name: str = "test", **inputs) -> ToolCall:
    return ToolCall(tool_use_id=tool_use_id, tool_name=tool_name, input=inputs)


def make_http_client_mock(response_text: str = "ok") -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------


class TestMockAdapter:
    async def test_returns_configured_text(self):
        adapter = MockAdapter(response="pong")
        result = await adapter.execute(make_call())
        assert result.content == "pong"
        assert result.is_error is False

    async def test_returns_configured_error(self):
        adapter = MockAdapter(response="boom", is_error=True)
        result = await adapter.execute(make_call())
        assert result.is_error is True
        assert result.content == "boom"

    async def test_propagates_tool_use_id(self):
        adapter = MockAdapter(response="ok")
        result = await adapter.execute(make_call(tool_use_id="tu-abc"))
        assert result.tool_use_id == "tu-abc"

    async def test_empty_response_default(self):
        adapter = MockAdapter()
        result = await adapter.execute(make_call())
        assert result.content == ""
        assert result.is_error is False

    async def test_not_error_by_default(self):
        adapter = MockAdapter(response="data")
        result = await adapter.execute(make_call())
        assert result.is_error is False


# ---------------------------------------------------------------------------
# HTTPAdapter
# ---------------------------------------------------------------------------


class TestHTTPAdapter:
    async def test_post_sends_json_body(self):
        mock_client = make_http_client_mock('{"result": "ok"}')
        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(url="http://example.com/api", method="POST")
            result = await adapter.execute(make_call(q="hello"))

        mock_client.post.assert_called_once()
        _, kwargs = mock_client.post.call_args
        assert kwargs["json"] == {"q": "hello"}
        assert result.content == '{"result": "ok"}'
        assert result.is_error is False

    async def test_get_sends_query_params(self):
        mock_client = make_http_client_mock("search results")
        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(url="http://example.com/search", method="GET")
            result = await adapter.execute(make_call(q="test"))

        mock_client.get.assert_called_once()
        _, kwargs = mock_client.get.call_args
        assert kwargs["params"] == {"q": "test"}
        assert result.content == "search results"

    async def test_method_case_insensitive(self):
        mock_client = make_http_client_mock("ok")
        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(url="http://example.com/api", method="post")
            await adapter.execute(make_call())
        mock_client.post.assert_called_once()

    async def test_http_error_returns_is_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("connection refused"))

        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(url="http://bad-host/api")
            result = await adapter.execute(make_call())

        assert result.is_error is True
        assert "connection refused" in result.content

    async def test_propagates_tool_use_id(self):
        mock_client = make_http_client_mock("ok")
        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(url="http://example.com/api")
            result = await adapter.execute(make_call(tool_use_id="tu-xyz"))
        assert result.tool_use_id == "tu-xyz"

    async def test_http_error_propagates_tool_use_id(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("err"))

        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(url="http://bad-host/api")
            result = await adapter.execute(make_call(tool_use_id="tu-err"))
        assert result.tool_use_id == "tu-err"

    async def test_custom_headers_passed_on_post(self):
        mock_client = make_http_client_mock("ok")
        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(
                url="http://example.com/api",
                headers={"Authorization": "Bearer token"},
            )
            await adapter.execute(make_call())

        _, kwargs = mock_client.post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer token"}

    async def test_default_method_is_post(self):
        mock_client = make_http_client_mock("ok")
        with patch("platform.tools.http_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = HTTPAdapter(url="http://example.com/api")
            await adapter.execute(make_call())
        mock_client.post.assert_called_once()
        mock_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# MCPAdapter
# ---------------------------------------------------------------------------


class TestMCPAdapter:
    def test_init_does_not_raise(self):
        from unittest.mock import MagicMock
        manager = MagicMock()
        MCPAdapter(server_command="npx", tool_name="read_file", _manager=manager)

    def test_init_stores_tool_name(self):
        from unittest.mock import MagicMock
        manager = MagicMock()
        adapter = MCPAdapter(server_command="npx", tool_name="read_file", _manager=manager)
        assert adapter._tool_name == "read_file"
