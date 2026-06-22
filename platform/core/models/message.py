"""Message and LLM response models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ToolUseContent(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = {}


class ToolResultContent(BaseModel):
    type: str = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    role: Role
    content: str | list[TextContent | ToolUseContent | ToolResultContent]


class LLMResponse(BaseModel):
    content: list[TextContent | ToolUseContent]
    stop_reason: StopReason
    input_tokens: int = 0
    output_tokens: int = 0
