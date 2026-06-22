"""MockAdapter — returns preconfigured responses for testing and demos."""

from __future__ import annotations

from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolCall, ToolResult


class MockAdapter(IToolAdapter):
    """Returns a static preconfigured response without making any external call.

    Configured via adapter_config in tools.yaml:
        response: "the mocked response string"
        is_error: false
    """

    def __init__(self, response: str = "", is_error: bool = False) -> None:
        self._response = response
        self._is_error = is_error

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_use_id=call.tool_use_id,
            content=self._response,
            is_error=self._is_error,
        )
