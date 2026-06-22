"""HTTPAdapter — executes tool calls via HTTP REST requests."""

from __future__ import annotations

import httpx

from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolCall, ToolResult


class HTTPAdapter(IToolAdapter):
    """Makes an HTTP request to a configured endpoint to execute a tool call.

    Configured via adapter_config in tools.yaml:
        url: "https://api.example.com/endpoint"
        method: "POST"
        headers: {}
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._url = url
        self._method = method.upper()
        self._headers = headers or {}

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            async with httpx.AsyncClient() as client:
                if self._method == "GET":
                    response = await client.get(
                        self._url,
                        params=call.input,
                        headers=self._headers,
                    )
                else:
                    response = await client.post(
                        self._url,
                        json=call.input,
                        headers=self._headers,
                    )
                response.raise_for_status()
                return ToolResult(tool_use_id=call.tool_use_id, content=response.text)
        except httpx.HTTPError as exc:
            return ToolResult(tool_use_id=call.tool_use_id, content=str(exc), is_error=True)
