"""Unit tests for GitHubAdapter."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from platform.core.models.tool import ToolCall
from platform.tools.github_adapter import GitHubAdapter


_PR_INPUT = {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}
_BASE = "https://api.github.com"


def make_call(**inputs) -> ToolCall:
    return ToolCall(tool_use_id="tu-1", tool_name="github_test", input=inputs or _PR_INPUT)


def make_github_client(response_text: str = '{"title": "Fix bug"}') -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client


def make_status_error_client(status_code: int = 404) -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=MagicMock()
        )
    )

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client


def make_network_error_client() -> MagicMock:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    return mock_client


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestGitHubAdapterInit:
    def test_unknown_operation_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported GitHub operation"):
            GitHubAdapter(operation="delete_repo")

    def test_known_operations_do_not_raise(self):
        for op in ("get_pull_request", "get_changed_files", "get_diff"):
            GitHubAdapter(operation=op)

    def test_reads_token_from_env(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}):
            adapter = GitHubAdapter(operation="get_pull_request")
        assert adapter._token == "ghp_test123"

    def test_empty_token_when_env_var_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            adapter = GitHubAdapter(operation="get_pull_request")
        assert adapter._token == ""


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestGitHubAdapterURLs:
    async def test_get_pull_request_url(self):
        mock_client = make_github_client()
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_pull_request")
            await adapter.execute(make_call(**_PR_INPUT))

        url = mock_client.get.call_args[0][0]
        assert url == f"{_BASE}/repos/octocat/Hello-World/pulls/42"

    async def test_get_changed_files_url(self):
        mock_client = make_github_client()
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_changed_files")
            await adapter.execute(make_call(**_PR_INPUT))

        url = mock_client.get.call_args[0][0]
        assert url == f"{_BASE}/repos/octocat/Hello-World/pulls/42/files"

    async def test_get_diff_url(self):
        mock_client = make_github_client()
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_diff")
            await adapter.execute(make_call(**_PR_INPUT))

        url = mock_client.get.call_args[0][0]
        assert url == f"{_BASE}/repos/octocat/Hello-World/pulls/42"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestGitHubAdapterHeaders:
    async def test_get_diff_uses_diff_accept_header(self):
        mock_client = make_github_client("diff content")
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_diff")
            await adapter.execute(make_call(**_PR_INPUT))

        headers = mock_client.get.call_args[1]["headers"]
        assert headers["Accept"] == "application/vnd.github.diff"

    async def test_get_pull_request_uses_json_accept_header(self):
        mock_client = make_github_client()
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_pull_request")
            await adapter.execute(make_call(**_PR_INPUT))

        headers = mock_client.get.call_args[1]["headers"]
        assert headers["Accept"] == "application/vnd.github+json"

    async def test_auth_header_included_when_token_set(self):
        mock_client = make_github_client()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_abc"}):
            adapter = GitHubAdapter(operation="get_pull_request")
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            await adapter.execute(make_call(**_PR_INPUT))

        headers = mock_client.get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer ghp_abc"

    async def test_no_auth_header_when_token_empty(self):
        mock_client = make_github_client()
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            adapter = GitHubAdapter(operation="get_pull_request")
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            await adapter.execute(make_call(**_PR_INPUT))

        headers = mock_client.get.call_args[1]["headers"]
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestGitHubAdapterSuccess:
    async def test_returns_response_text(self):
        mock_client = make_github_client('{"title": "Add feature"}')
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_pull_request")
            result = await adapter.execute(make_call(**_PR_INPUT))

        assert result.content == '{"title": "Add feature"}'
        assert result.is_error is False

    async def test_propagates_tool_use_id(self):
        mock_client = make_github_client()
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_pull_request")
            call = ToolCall(tool_use_id="tu-xyz", tool_name="github_get_pr", input=_PR_INPUT)
            result = await adapter.execute(call)

        assert result.tool_use_id == "tu-xyz"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestGitHubAdapterErrors:
    async def test_http_status_error_returns_is_error(self):
        mock_client = make_status_error_client(404)
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_pull_request")
            result = await adapter.execute(make_call(**_PR_INPUT))

        assert result.is_error is True
        assert result.content != ""

    async def test_network_error_returns_is_error(self):
        mock_client = make_network_error_client()
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_pull_request")
            result = await adapter.execute(make_call(**_PR_INPUT))

        assert result.is_error is True
        assert "connection refused" in result.content

    async def test_propagates_tool_use_id_on_error(self):
        mock_client = make_status_error_client(401)
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            adapter = GitHubAdapter(operation="get_pull_request")
            call = ToolCall(tool_use_id="tu-err", tool_name="github_get_pr", input=_PR_INPUT)
            result = await adapter.execute(call)

        assert result.tool_use_id == "tu-err"
        assert result.is_error is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestGitHubAdapterValidation:
    """Adapter must reject calls missing required fields without making HTTP requests."""

    async def _execute_with_input(self, input: dict) -> object:
        adapter = GitHubAdapter(operation="get_pull_request")
        call = ToolCall(tool_use_id="tu-v", tool_name="github_get_pr", input=input)
        return await adapter.execute(call)

    async def test_missing_owner_returns_is_error(self):
        result = await self._execute_with_input({"repo": "Hello-World", "pull_number": 42})
        assert result.is_error is True
        assert "owner" in result.content

    async def test_missing_repo_returns_is_error(self):
        result = await self._execute_with_input({"owner": "octocat", "pull_number": 42})
        assert result.is_error is True
        assert "repo" in result.content

    async def test_missing_pull_number_returns_is_error(self):
        result = await self._execute_with_input({"owner": "octocat", "repo": "Hello-World"})
        assert result.is_error is True
        assert "pull_number" in result.content

    async def test_empty_owner_returns_is_error(self):
        result = await self._execute_with_input({"owner": "", "repo": "Hello-World", "pull_number": 42})
        assert result.is_error is True

    async def test_empty_input_returns_is_error(self):
        result = await self._execute_with_input({})
        assert result.is_error is True

    async def test_validation_error_propagates_tool_use_id(self):
        adapter = GitHubAdapter(operation="get_pull_request")
        call = ToolCall(tool_use_id="tu-val", tool_name="github_get_pr", input={"owner": "octocat"})
        result = await adapter.execute(call)
        assert result.tool_use_id == "tu-val"
        assert result.is_error is True

    async def test_all_fields_present_does_not_trigger_validation_error(self):
        mock_client = make_github_client()
        with patch("platform.tools.github_adapter.httpx.AsyncClient", return_value=mock_client):
            result = await self._execute_with_input(_PR_INPUT)
        assert result.is_error is False
