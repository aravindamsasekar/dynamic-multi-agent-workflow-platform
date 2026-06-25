"""AgentRuntime — executes a single agent to completion."""

from __future__ import annotations

from platform.core.exceptions import PatternExecutionError
from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.agent import AgentDefinition, AgentResult
from platform.core.models.context import ExecutionContext
from platform.core.models.events import (
    AgentCalledEvent,
    AgentCompletedEvent,
    ToolCalledEvent,
    ToolCompletedEvent,
)
from platform.core.models.message import (
    Message,
    Role,
    StopReason,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from platform.core.models.tool import ToolCall, ToolDefinition
from platform.policy.engine import HookPoint

_MAX_TOOL_ROUNDS = 10


class AgentRuntime:
    """Runs a single agent: LLM call → tool loop → final answer.

    Shared by all pattern executors. Each call to run() is independent
    and scoped to a single agent turn within a workflow run.
    """

    def __init__(self, llm_provider: ILLMProvider, context: ExecutionContext) -> None:
        self._llm_provider = llm_provider
        self._tool_registry = context.tool_registry
        self._memory_store = context.memory_store
        self._policy_engine = context.policy_engine
        self._observer = context.observer
        self._run_id = context.run_id

    async def run(self, agent_def: AgentDefinition, input: str) -> AgentResult:
        """Execute agent until a final text response is returned.

        Steps:
        1.  Load conversation history from memory_store.
        2.  Build LLM message list: system message + history + user input.
        3.  Resolve ToolDefinition objects for this agent's tool_names.
        4.  Evaluate policy at PRE_AGENT hook.
        5.  Emit AgentCalledEvent.
        6.  Call LLMProvider.complete(); if TOOL_USE: PRE_TOOL → execute →
            POST_TOOL → append result → loop until END_TURN.
        7.  Evaluate policy at POST_AGENT hook.
        8.  Persist all new messages to memory_store.
        9.  Emit AgentCompletedEvent.
        10. Return AgentResult.
        """
        # --- 1. Load history ---
        history = self._memory_store.get_history(self._run_id, agent_def.agent_id)

        # --- 2. Build message list ---
        llm_messages: list[Message] = []
        if agent_def.system_prompt:
            llm_messages.append(Message(role=Role.SYSTEM, content=agent_def.system_prompt))
        llm_messages.extend(history)

        user_message = Message(role=Role.USER, content=input)
        llm_messages.append(user_message)

        # new_messages: only what is generated during this call (persisted at the end)
        new_messages: list[Message] = [user_message]

        # --- 3. Resolve tool definitions ---
        tool_defs: list[ToolDefinition] = []
        for name in agent_def.tool_names:
            td = self._tool_registry.get_definition(name)
            if td is not None:
                tool_defs.append(td)

        # --- 4. PRE_AGENT policy ---
        self._policy_engine.evaluate(
            HookPoint.PRE_AGENT, {"agent_id": agent_def.agent_id, "input": input}
        )

        # --- 5. Emit AgentCalledEvent ---
        self._observer.on_event(
            AgentCalledEvent(run_id=self._run_id, data={"agent_id": agent_def.agent_id})
        )

        # --- 6. LLM → tool loop ---
        tool_calls_made = 0
        tools_arg = tool_defs if tool_defs else None
        response = await self._llm_provider.complete(llm_messages, tools=tools_arg)
        tool_round = 0

        while response.stop_reason == StopReason.TOOL_USE:
            if tool_round >= _MAX_TOOL_ROUNDS:
                raise PatternExecutionError(
                    f"Agent '{agent_def.agent_id}' exceeded {_MAX_TOOL_ROUNDS} tool-call rounds"
                )
            tool_round += 1

            # Append assistant message containing tool-use blocks
            assistant_msg = Message(role=Role.ASSISTANT, content=list(response.content))
            llm_messages.append(assistant_msg)
            new_messages.append(assistant_msg)

            for tc in (c for c in response.content if isinstance(c, ToolUseContent)):
                tool_calls_made += 1
                tool_call = ToolCall(
                    tool_use_id=tc.id, tool_name=tc.name, input=tc.input
                )

                self._policy_engine.evaluate(
                    HookPoint.PRE_TOOL, {"tool_name": tc.name, "input": tc.input}
                )
                self._observer.on_event(
                    ToolCalledEvent(
                        run_id=self._run_id,
                        data={"tool_name": tc.name, "input": tc.input},
                    )
                )

                adapter = self._tool_registry.get(tc.name)
                result = await adapter.execute(tool_call)

                self._policy_engine.evaluate(
                    HookPoint.POST_TOOL, {"tool_name": tc.name, "result": result.content}
                )
                self._observer.on_event(
                    ToolCompletedEvent(
                        run_id=self._run_id,
                        data={
                            "tool_name": tc.name,
                            "tool_input": tc.input,
                            "result": result.content,
                            "is_error": result.is_error,
                        },
                    )
                )

                tool_result_msg = Message(
                    role=Role.USER,
                    content=[
                        ToolResultContent(
                            tool_use_id=result.tool_use_id,
                            content=result.content,
                            is_error=result.is_error,
                        )
                    ],
                )
                llm_messages.append(tool_result_msg)
                new_messages.append(tool_result_msg)

            response = await self._llm_provider.complete(llm_messages, tools=tools_arg)

        # --- 7. POST_AGENT policy ---
        final_text = next(
            (c.text for c in response.content if isinstance(c, TextContent)), ""
        )
        self._policy_engine.evaluate(
            HookPoint.POST_AGENT, {"agent_id": agent_def.agent_id, "output": final_text}
        )

        # --- 8. Persist new messages ---
        final_assistant_msg = Message(role=Role.ASSISTANT, content=list(response.content))
        new_messages.append(final_assistant_msg)
        for msg in new_messages:
            self._memory_store.append(self._run_id, agent_def.agent_id, msg)

        # --- 9. Emit AgentCompletedEvent ---
        self._observer.on_event(
            AgentCompletedEvent(
                run_id=self._run_id,
                data={"agent_id": agent_def.agent_id, "output": final_text},
            )
        )

        # --- 10. Return ---
        return AgentResult(
            agent_id=agent_def.agent_id,
            output=final_text,
            tool_calls_made=tool_calls_made,
        )
