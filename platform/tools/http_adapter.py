"""HTTPAdapter — executes tool calls via HTTP REST requests."""

from __future__ import annotations

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
        self._method = method
        self._headers = headers or {}
        # TODO: initialize httpx.AsyncClient

    async def execute(self, call: ToolCall) -> ToolResult:
        # TODO: implement
        raise NotImplementedError
