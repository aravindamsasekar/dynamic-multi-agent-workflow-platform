"""AgentRuntime — executes a single agent to completion."""

from __future__ import annotations

from platform.core.models.agent import AgentDefinition, AgentResult
from platform.core.models.context import ExecutionContext


class AgentRuntime:
    """Runs a single agent: LLM call → tool loop → final answer.

    Shared by all pattern executors. Each call to run() is independent
    and scoped to a single agent turn within a workflow run.
    """

    def __init__(self, context: ExecutionContext) -> None:
        self._context = context
        # TODO: extract llm_provider, tool_registry, memory_store, policy_engine, observer

    async def run(self, agent_def: AgentDefinition, input: str) -> AgentResult:
        """Execute agent until a final text response is returned.

        Steps:
        1. Load conversation history from memory_store
        2. Evaluate policy at PRE_AGENT hook
        3. Emit AgentCalledEvent to observer
        4. Call LLMProvider.complete() with messages and agent tools
        5. If tool_use blocks: evaluate PRE_TOOL, execute via tool_registry,
           evaluate POST_TOOL, append ToolResult to messages, loop
        6. Evaluate policy at POST_AGENT hook
        7. Save updated history to memory_store
        8. Emit AgentCompletedEvent to observer
        9. Return AgentResult
        """
        # TODO: implement
        raise NotImplementedError
