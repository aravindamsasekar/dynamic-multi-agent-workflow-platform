"""OpenAIProvider — OpenAI implementation of ILLMProvider."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.message import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from platform.core.models.tool import ToolDefinition

load_dotenv()

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
}


class OpenAIProvider(ILLMProvider):
    """OpenAI implementation of ILLMProvider.

    API key resolution order: constructor arg → OPENAI_API_KEY env var.
    Model resolution order: constructor arg → OPENAI_MODEL env var → gpt-4o-mini.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        _client: AsyncOpenAI | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY in the environment or .env file."
            )
        self._model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self._client = _client or AsyncOpenAI(api_key=resolved_key)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_openai_message(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        return _to_llm_response(response.choices[0], response.usage)


def _to_openai_message(message: Message) -> ChatCompletionMessageParam:
    if isinstance(message.content, str):
        return {"role": message.role.value, "content": message.content}  # type: ignore[return-value]

    parts = message.content

    if message.role == Role.USER:
        if parts and isinstance(parts[0], ToolResultContent):
            result = parts[0]
            return {  # type: ignore[return-value]
                "role": "tool",
                "tool_call_id": result.tool_use_id,
                "content": result.content,
            }
        text = " ".join(c.text for c in parts if isinstance(c, TextContent))
        return {"role": "user", "content": text}  # type: ignore[return-value]

    # role == ASSISTANT
    text_parts = [c for c in parts if isinstance(c, TextContent)]
    tool_use_parts = [c for c in parts if isinstance(c, ToolUseContent)]

    oai_msg: dict[str, Any] = {
        "role": "assistant",
        "content": " ".join(t.text for t in text_parts) if text_parts else None,
    }
    if tool_use_parts:
        oai_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
            }
            for tc in tool_use_parts
        ]
    return oai_msg  # type: ignore[return-value]


def _to_openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _to_llm_response(choice: Any, usage: Any) -> LLMResponse:
    message = choice.message
    content: list[TextContent | ToolUseContent] = []

    if message.content:
        content.append(TextContent(text=message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            content.append(
                ToolUseContent(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                )
            )

    return LLMResponse(
        content=content,
        stop_reason=_STOP_REASON_MAP.get(choice.finish_reason, StopReason.END_TURN),
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
