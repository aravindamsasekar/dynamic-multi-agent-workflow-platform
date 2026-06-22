"""Unit tests for MockLLMProvider and OpenAIProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from platform.core.models.message import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    TextContent,
    ToolUseContent,
)
from platform.core.models.tool import AdapterType, ToolDefinition
from platform.llm.mock_provider import MockLLMProvider
from platform.llm.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_text_response(text: str = "hello") -> LLMResponse:
    return LLMResponse(
        content=[TextContent(text=text)],
        stop_reason=StopReason.END_TURN,
    )


def make_tool_response(tool_id: str = "tc-1", tool_name: str = "search") -> LLMResponse:
    return LLMResponse(
        content=[ToolUseContent(id=tool_id, name=tool_name, input={"q": "test"})],
        stop_reason=StopReason.TOOL_USE,
    )


def make_messages() -> list[Message]:
    return [Message(role=Role.USER, content="What is 2+2?")]


def make_tool_def() -> ToolDefinition:
    return ToolDefinition(
        name="search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        adapter_type=AdapterType.HTTP,
    )


def make_oai_completion(text: str = "ok", finish_reason: str = "stop") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


def make_oai_tool_completion(tool_id: str = "tc-1", tool_name: str = "search") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 15

    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = tool_name
    tc.function.arguments = '{"q": "test"}'

    message = MagicMock()
    message.content = None
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


def make_mock_openai_client(completion: MagicMock) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------


class TestMockLLMProvider:
    async def test_returns_text_response(self):
        provider = MockLLMProvider(responses=[make_text_response("hi there")])
        result = await provider.complete(make_messages())
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "hi there"
        assert result.stop_reason == StopReason.END_TURN

    async def test_returns_tool_call_response(self):
        provider = MockLLMProvider(responses=[make_tool_response()])
        result = await provider.complete(make_messages(), tools=[make_tool_def()])
        assert result.stop_reason == StopReason.TOOL_USE
        assert isinstance(result.content[0], ToolUseContent)
        assert result.content[0].name == "search"
        assert result.content[0].input == {"q": "test"}

    async def test_responses_consumed_in_order(self):
        provider = MockLLMProvider(
            responses=[make_text_response("first"), make_text_response("second")]
        )
        first = await provider.complete(make_messages())
        second = await provider.complete(make_messages())
        assert first.content[0].text == "first"
        assert second.content[0].text == "second"

    async def test_raises_when_queue_exhausted(self):
        provider = MockLLMProvider(responses=[make_text_response()])
        await provider.complete(make_messages())
        with pytest.raises(RuntimeError, match="exhausted"):
            await provider.complete(make_messages())

    async def test_tools_argument_accepted_but_ignored(self):
        provider = MockLLMProvider(responses=[make_text_response()])
        result = await provider.complete(make_messages(), tools=[make_tool_def()])
        assert result.stop_reason == StopReason.END_TURN

    async def test_empty_responses_raises_immediately(self):
        provider = MockLLMProvider(responses=[])
        with pytest.raises(RuntimeError, match="exhausted"):
            await provider.complete(make_messages())


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_init_with_explicit_api_key(self):
        provider = OpenAIProvider(api_key="sk-test", _client=MagicMock())
        assert provider._model == "gpt-4o-mini"

    def test_init_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        provider = OpenAIProvider(_client=MagicMock())
        assert provider._model == "gpt-4o-mini"

    def test_init_reads_model_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        provider = OpenAIProvider(_client=MagicMock())
        assert provider._model == "gpt-4o"

    def test_explicit_model_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        provider = OpenAIProvider(model="gpt-4o-mini", _client=MagicMock())
        assert provider._model == "gpt-4o-mini"

    def test_init_raises_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider()

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        provider = OpenAIProvider(api_key="sk-explicit", _client=MagicMock())
        assert provider is not None

    async def test_complete_returns_text_response(self):
        client = make_mock_openai_client(make_oai_completion("hello"))
        provider = OpenAIProvider(api_key="sk-test", _client=client)
        result = await provider.complete(make_messages())
        assert result.stop_reason == StopReason.END_TURN
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "hello"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    async def test_complete_returns_tool_call_response(self):
        client = make_mock_openai_client(make_oai_tool_completion())
        provider = OpenAIProvider(api_key="sk-test", _client=client)
        result = await provider.complete(make_messages(), tools=[make_tool_def()])
        assert result.stop_reason == StopReason.TOOL_USE
        assert isinstance(result.content[0], ToolUseContent)
        assert result.content[0].name == "search"
        assert result.content[0].input == {"q": "test"}
        assert result.content[0].id == "tc-1"

    async def test_complete_passes_tools_to_openai(self):
        client = make_mock_openai_client(make_oai_completion())
        provider = OpenAIProvider(api_key="sk-test", _client=client)
        await provider.complete(make_messages(), tools=[make_tool_def()])
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["function"]["name"] == "search"
        assert call_kwargs["tool_choice"] == "auto"

    async def test_complete_no_tools_omits_tools_param(self):
        client = make_mock_openai_client(make_oai_completion())
        provider = OpenAIProvider(api_key="sk-test", _client=client)
        await provider.complete(make_messages())
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    async def test_complete_maps_length_finish_reason(self):
        client = make_mock_openai_client(make_oai_completion("truncated", "length"))
        provider = OpenAIProvider(api_key="sk-test", _client=client)
        result = await provider.complete(make_messages())
        assert result.stop_reason == StopReason.MAX_TOKENS

    async def test_complete_sends_correct_model(self):
        client = make_mock_openai_client(make_oai_completion())
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o", _client=client)
        await provider.complete(make_messages())
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
