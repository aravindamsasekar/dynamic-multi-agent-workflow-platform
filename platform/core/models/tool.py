"""Tool-related models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class AdapterType(str, Enum):
    MOCK = "mock"
    HTTP = "http"
    MCP = "mcp"


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    adapter_type: AdapterType
    adapter_config: dict[str, Any] = {}


class ToolCall(BaseModel):
    tool_use_id: str
    tool_name: str
    input: dict[str, Any] = {}


class ToolResult(BaseModel):
    tool_use_id: str
    content: str
    is_error: bool = False
