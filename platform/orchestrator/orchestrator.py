"""Orchestrator — entry point for workflow execution."""

from __future__ import annotations

from platform.core.models.context import ExecutionContext
from platform.core.models.workflow import WorkflowResult


class Orchestrator:
    """Selects the correct pattern executor and drives workflow execution.

    Callers only need: orchestrator.run(workflow_id, input) -> WorkflowResult.
    The Orchestrator is stateless per invocation; all run state lives in RunManager.
    """

    def __init__(
        self,
        workflow_registry: object,
        agent_registry: object,
        tool_registry: object,
        memory_store: object,
        policy_engine: object,
        observer: object,
        run_manager: object,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._memory_store = memory_store
        self._policy_engine = policy_engine
        self._observer = observer
        self._run_manager = run_manager
        # TODO: build PatternType → PatternExecutor dispatch map

    async def run(self, workflow_id: str, input: str) -> WorkflowResult:
        """Execute a workflow by id.

        Steps:
        1. Lookup WorkflowDefinition from workflow_registry
        2. Create WorkflowRun via run_manager (status: PENDING → RUNNING)
        3. Emit WorkflowStartedEvent to observer
        4. Build ExecutionContext
        5. Select PatternExecutor by workflow_definition.pattern
        6. Call PatternExecutor.execute(context)
        7. Mark run COMPLETED via run_manager, emit WorkflowCompletedEvent
        8. Return WorkflowResult
        """
        # TODO: implement
        raise NotImplementedError

    def _build_context(self, run_id: str, workflow_definition: object) -> ExecutionContext:
        # TODO: implement
        raise NotImplementedError
