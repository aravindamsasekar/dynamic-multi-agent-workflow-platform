"""Unit tests for AgentRuntime."""

from __future__ import annotations

from typing import Any

import pytest

from platform.agent.runtime import AgentRuntime
from platform.core.exceptions import PatternExecutionError, PolicyViolation
from platform.core.interfaces.llm import ILLMProvider
from platform.core.interfaces.observer import IObserver
from platform.core.interfaces.policy import IRule, PolicyDecision
from platform.core.models.agent import AgentDefinition, AgentResult
from platform.core.models.context import ExecutionContext
from platform.core.models.events import (
    AgentCalledEvent,
    AgentCompletedEvent,
    ToolCalledEvent,
    ToolCompletedEvent,
    WorkflowEvent,
)
from platform.core.models.message import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    TextContent,
    ToolUseContent,
)
from platform.core.models.tool import AdapterType, ToolCall, ToolDefinition, ToolResult
from platform.core.models.workflow import PatternType, WorkflowDefinition
from platform.llm.mock_provider import MockLLMProvider
from platform.memory.in_memory_store import InMemoryStore
from platform.policy.engine import PolicyEngine
from platform.registries.tool_registry import ToolRegistry
from platform.tools.mock_adapter import MockAdapter


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------


class _CapturingObserver(IObserver):
    def __init__(self) -> None:
        self.events: list[WorkflowEvent] = []

    def on_event(self, event: WorkflowEvent) -> None:
        self.events.append(event)


class _SpyLLMProvider(ILLMProvider):
    """Wraps MockLLMProvider and records every call's message list."""

    def __init__(self, inner: ILLMProvider) -> None:
        self._inner = inner
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return await self._inner.complete(messages, tools)


class _BlockingRule(IRule):
    def check(self, context: dict[str, Any]) -> PolicyDecision:
        return PolicyDecision.BLOCK


class _BlockWhenKeyPresentRule(IRule):
    """Blocks only when a specific key appears in the evaluation context."""

    def __init__(self, key: str) -> None:
        self._key = key

    def check(self, context: dict[str, Any]) -> PolicyDecision:
        return PolicyDecision.BLOCK if self._key in context else PolicyDecision.ALLOW


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_RUN_ID = "run-test"
_AGENT_ID = "agent-1"


def make_agent_def(
    tool_names: list[str] | None = None,
    system_prompt: str = "You are a test agent.",
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=_AGENT_ID,
        name="Test Agent",
        system_prompt=system_prompt,
        tool_names=tool_names or [],
    )


def make_tool_def(name: str = "search") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Search the web",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        adapter_type=AdapterType.MOCK,
    )


def make_text_response(text: str = "final answer") -> LLMResponse:
    return LLMResponse(
        content=[TextContent(text=text)],
        stop_reason=StopReason.END_TURN,
    )


def make_tool_response(
    tool_id: str = "tc-1",
    tool_name: str = "search",
    tool_input: dict | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=[ToolUseContent(id=tool_id, name=tool_name, input=tool_input or {"q": "test"})],
        stop_reason=StopReason.TOOL_USE,
    )


def make_context(
    tool_registry: ToolRegistry | None = None,
    memory_store: InMemoryStore | None = None,
    policy_engine: PolicyEngine | None = None,
    observer: _CapturingObserver | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        run_id=_RUN_ID,
        workflow_definition=WorkflowDefinition(
            workflow_id="wf-1",
            name="Test Workflow",
            pattern=PatternType.ROUTER,
        ),
        shared_state=None,
        workflow_registry=None,
        agent_registry=None,
        tool_registry=tool_registry or ToolRegistry(),
        memory_store=memory_store or InMemoryStore(),
        policy_engine=policy_engine or PolicyEngine(),
        observer=observer or _CapturingObserver(),
    )


def make_runtime(
    provider: ILLMProvider,
    context: ExecutionContext,
) -> AgentRuntime:
    return AgentRuntime(llm_provider=provider, context=context)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentRuntime:
    async def test_run_returns_text_response(self):
        provider = MockLLMProvider(responses=[make_text_response("hello world")])
        runtime = make_runtime(provider, make_context())

        result = await runtime.run(make_agent_def(), "say hello")

        assert isinstance(result, AgentResult)
        assert result.agent_id == _AGENT_ID
        assert result.output == "hello world"
        assert result.tool_calls_made == 0

    async def test_run_with_single_tool_call(self):
        tool_def = make_tool_def("search")
        adapter = MockAdapter(response="Paris")
        registry = ToolRegistry()
        registry.register("search", adapter, tool_def=tool_def)

        provider = MockLLMProvider(
            responses=[
                make_tool_response("tc-1", "search"),
                make_text_response("The capital is Paris."),
            ]
        )
        runtime = make_runtime(provider, make_context(tool_registry=registry))

        result = await runtime.run(make_agent_def(tool_names=["search"]), "capital of France?")

        assert result.output == "The capital is Paris."
        assert result.tool_calls_made == 1

    async def test_run_with_multiple_sequential_tool_calls(self):
        tool_def = make_tool_def("lookup")
        adapter = MockAdapter(response="42")
        registry = ToolRegistry()
        registry.register("lookup", adapter, tool_def=tool_def)

        provider = MockLLMProvider(
            responses=[
                make_tool_response("tc-1", "lookup"),
                make_tool_response("tc-2", "lookup"),
                make_text_response("Done."),
            ]
        )
        runtime = make_runtime(provider, make_context(tool_registry=registry))

        result = await runtime.run(make_agent_def(tool_names=["lookup"]), "look up twice")

        assert result.tool_calls_made == 2
        assert result.output == "Done."

    async def test_run_raises_policy_violation_at_pre_agent(self):
        engine = PolicyEngine(rules=[_BlockingRule()])
        observer = _CapturingObserver()
        provider = MockLLMProvider(responses=[make_text_response()])
        runtime = make_runtime(provider, make_context(policy_engine=engine, observer=observer))

        with pytest.raises(PolicyViolation):
            await runtime.run(make_agent_def(), "input")

        # PRE_AGENT blocks before AgentCalledEvent is emitted
        assert not any(isinstance(e, AgentCalledEvent) for e in observer.events)

    async def test_run_raises_policy_violation_at_pre_tool(self):
        tool_def = make_tool_def("search")
        registry = ToolRegistry()
        registry.register("search", MockAdapter(response="ok"), tool_def=tool_def)

        # blocks when "tool_name" key is present (i.e. at PRE_TOOL), passes at PRE_AGENT
        engine = PolicyEngine(rules=[_BlockWhenKeyPresentRule("tool_name")])
        observer = _CapturingObserver()

        provider = MockLLMProvider(
            responses=[make_tool_response("tc-1", "search"), make_text_response()]
        )
        runtime = make_runtime(
            provider, make_context(tool_registry=registry, policy_engine=engine, observer=observer)
        )

        with pytest.raises(PolicyViolation):
            await runtime.run(make_agent_def(tool_names=["search"]), "search something")

        assert any(isinstance(e, AgentCalledEvent) for e in observer.events)
        assert not any(isinstance(e, AgentCompletedEvent) for e in observer.events)

    async def test_run_raises_policy_violation_at_post_agent(self):
        # blocks when "output" key is present (i.e. at POST_AGENT)
        engine = PolicyEngine(rules=[_BlockWhenKeyPresentRule("output")])
        observer = _CapturingObserver()
        provider = MockLLMProvider(responses=[make_text_response()])
        runtime = make_runtime(provider, make_context(policy_engine=engine, observer=observer))

        with pytest.raises(PolicyViolation):
            await runtime.run(make_agent_def(), "input")

        assert any(isinstance(e, AgentCalledEvent) for e in observer.events)
        assert not any(isinstance(e, AgentCompletedEvent) for e in observer.events)

    async def test_run_emits_agent_called_event(self):
        observer = _CapturingObserver()
        provider = MockLLMProvider(responses=[make_text_response()])
        runtime = make_runtime(provider, make_context(observer=observer))

        await runtime.run(make_agent_def(), "hi")

        called_events = [e for e in observer.events if isinstance(e, AgentCalledEvent)]
        assert len(called_events) == 1
        assert called_events[0].run_id == _RUN_ID
        assert called_events[0].data["agent_id"] == _AGENT_ID

    async def test_run_emits_agent_completed_event(self):
        observer = _CapturingObserver()
        provider = MockLLMProvider(responses=[make_text_response("result")])
        runtime = make_runtime(provider, make_context(observer=observer))

        await runtime.run(make_agent_def(), "hi")

        completed_events = [e for e in observer.events if isinstance(e, AgentCompletedEvent)]
        assert len(completed_events) == 1
        assert completed_events[0].data["agent_id"] == _AGENT_ID
        assert completed_events[0].data["output"] == "result"

    async def test_run_emits_tool_called_and_completed_events(self):
        tool_def = make_tool_def("search")
        registry = ToolRegistry()
        registry.register("search", MockAdapter(response="result"), tool_def=tool_def)

        observer = _CapturingObserver()
        provider = MockLLMProvider(
            responses=[make_tool_response("tc-1", "search"), make_text_response()]
        )
        runtime = make_runtime(
            provider, make_context(tool_registry=registry, observer=observer)
        )

        await runtime.run(make_agent_def(tool_names=["search"]), "search query")

        assert any(isinstance(e, ToolCalledEvent) for e in observer.events)
        assert any(isinstance(e, ToolCompletedEvent) for e in observer.events)

        tool_called = next(e for e in observer.events if isinstance(e, ToolCalledEvent))
        assert tool_called.data["tool_name"] == "search"

    async def test_run_saves_messages_to_memory_store(self):
        memory = InMemoryStore()
        provider = MockLLMProvider(responses=[make_text_response("stored")])
        runtime = make_runtime(provider, make_context(memory_store=memory))

        await runtime.run(make_agent_def(), "my input")

        history = memory.get_history(_RUN_ID, _AGENT_ID)
        roles = [m.role for m in history]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

        user_msg = next(m for m in history if m.role == Role.USER)
        assert user_msg.content == "my input"

        assistant_msg = next(m for m in history if m.role == Role.ASSISTANT)
        assert isinstance(assistant_msg.content, list)
        assert assistant_msg.content[0].text == "stored"

    async def test_run_saves_tool_messages_to_memory(self):
        tool_def = make_tool_def("search")
        registry = ToolRegistry()
        registry.register("search", MockAdapter(response="42"), tool_def=tool_def)

        memory = InMemoryStore()
        provider = MockLLMProvider(
            responses=[make_tool_response("tc-1", "search"), make_text_response("answer")]
        )
        runtime = make_runtime(
            provider, make_context(tool_registry=registry, memory_store=memory)
        )

        await runtime.run(make_agent_def(tool_names=["search"]), "question")

        history = memory.get_history(_RUN_ID, _AGENT_ID)
        # user input, assistant tool-use, user tool-result, assistant final
        assert len(history) == 4
        assert history[0].role == Role.USER
        assert history[1].role == Role.ASSISTANT  # tool-use
        assert history[2].role == Role.USER       # tool-result
        assert history[3].role == Role.ASSISTANT  # final answer

    async def test_run_loads_existing_history(self):
        memory = InMemoryStore()
        prior_user = Message(role=Role.USER, content="prior input")
        prior_assistant = Message(role=Role.ASSISTANT, content=[TextContent(text="prior response")])
        memory.append(_RUN_ID, _AGENT_ID, prior_user)
        memory.append(_RUN_ID, _AGENT_ID, prior_assistant)

        inner = MockLLMProvider(responses=[make_text_response()])
        spy = _SpyLLMProvider(inner)
        runtime = make_runtime(spy, make_context(memory_store=memory))

        await runtime.run(make_agent_def(), "new input")

        first_call = spy.calls[0]
        # system + prior_user + prior_assistant + new user = at least 4 messages
        message_contents = [m.content for m in first_call]
        assert prior_user.content in message_contents
        assert any(
            isinstance(m.content, list) and m.content[0].text == "prior response"
            for m in first_call
            if isinstance(m.content, list)
        )

    async def test_run_system_prompt_prepended_as_first_message(self):
        inner = MockLLMProvider(responses=[make_text_response()])
        spy = _SpyLLMProvider(inner)
        runtime = make_runtime(spy, make_context())

        agent = make_agent_def(system_prompt="You are a helpful assistant.")
        await runtime.run(agent, "hello")

        first_msg = spy.calls[0][0]
        assert first_msg.role == Role.SYSTEM
        assert first_msg.content == "You are a helpful assistant."

    async def test_run_tool_calls_made_count(self):
        tool_def = make_tool_def("tool")
        registry = ToolRegistry()
        registry.register("tool", MockAdapter(response="r"), tool_def=tool_def)

        provider = MockLLMProvider(
            responses=[
                make_tool_response("tc-1", "tool"),
                make_tool_response("tc-2", "tool"),
                make_tool_response("tc-3", "tool"),
                make_text_response("done"),
            ]
        )
        runtime = make_runtime(provider, make_context(tool_registry=registry))

        result = await runtime.run(make_agent_def(tool_names=["tool"]), "go")

        assert result.tool_calls_made == 3

    async def test_run_tool_not_found_propagates(self):
        # agent declares tool "missing" but registry has nothing registered
        provider = MockLLMProvider(responses=[make_tool_response("tc-1", "missing")])
        runtime = make_runtime(provider, make_context())

        from platform.core.exceptions import ToolNotFound

        with pytest.raises(ToolNotFound):
            await runtime.run(make_agent_def(tool_names=["missing"]), "go")
