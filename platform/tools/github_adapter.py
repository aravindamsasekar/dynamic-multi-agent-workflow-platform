"""GitHubAdapter — executes GitHub REST API tool calls."""

from __future__ import annotations

import os

import httpx

from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolCall, ToolResult

_BASE_URL = "https://api.github.com"
_SUPPORTED_OPERATIONS = frozenset({"get_pull_request", "get_changed_files", "get_diff"})


class GitHubAdapter(IToolAdapter):
    """Calls the GitHub REST API to fetch pull request data.

    Configured via adapter_config in tools.yaml:
        operation: get_pull_request | get_changed_files | get_diff

    ToolCall.input must contain:
        owner:       str  — repository owner or organization
        repo:        str  — repository name
        pull_number: int  — pull request number

    Authentication: reads GITHUB_TOKEN from environment at construction time.
    If GITHUB_TOKEN is not set, requests proceed unauthenticated (public repos only).
    """

    def __init__(self, operation: str) -> None:
        if operation not in _SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported GitHub operation '{operation}'. "
                f"Must be one of: {sorted(_SUPPORTED_OPERATIONS)}"
            )
        self._operation = operation
        self._token = os.getenv("GITHUB_TOKEN", "")

    async def execute(self, call: ToolCall) -> ToolResult:
        owner = call.input.get("owner")
        repo = call.input.get("repo")
        pull_number = call.input.get("pull_number")

        missing = []
        if not owner:
            missing.append("owner")
        if not repo:
            missing.append("repo")
        if pull_number is None:
            missing.append("pull_number")
        if missing:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=(
                    f"Missing required fields: {missing}. "
                    "Call this tool with owner (str), repo (str), and pull_number (int)."
                ),
                is_error=True,
            )

        headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        if self._operation == "get_diff":
            headers["Accept"] = "application/vnd.github.diff"
            url = f"{_BASE_URL}/repos/{owner}/{repo}/pulls/{pull_number}"
        elif self._operation == "get_changed_files":
            url = f"{_BASE_URL}/repos/{owner}/{repo}/pulls/{pull_number}/files"
        else:  # get_pull_request
            url = f"{_BASE_URL}/repos/{owner}/{repo}/pulls/{pull_number}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return ToolResult(tool_use_id=call.tool_use_id, content=response.text)
        except httpx.HTTPStatusError as exc:
            return ToolResult(tool_use_id=call.tool_use_id, content=str(exc), is_error=True)
        except httpx.HTTPError as exc:
            return ToolResult(tool_use_id=call.tool_use_id, content=str(exc), is_error=True)
