"""Orchestrator — entry point for workflow execution."""

from __future__ import annotations

import json
from typing import Any

from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.context import ExecutionContext
from platform.core.models.events import (
    WorkflowCompletedEvent,
    WorkflowFailedEvent,
    WorkflowStartedEvent,
)
from platform.core.models.workflow import PatternType, RunStatus, WorkflowDefinition, WorkflowResult
from platform.patterns.parallel_specialist import ParallelSpecialistExecutor
from platform.patterns.planner_executor_observer import PlannerExecutorObserverExecutor
from platform.patterns.router import RouterExecutor


class Orchestrator:
    """Selects the correct pattern executor and drives workflow execution.

    Callers only need: orchestrator.run(workflow_id, input) -> WorkflowResult.
    All run state lives in RunManager; SharedState is injected, not constructed here.
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
        llm_provider: ILLMProvider,
        shared_state: object,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._memory_store = memory_store
        self._policy_engine = policy_engine
        self._observer = observer
        self._run_manager = run_manager
        self._shared_state = shared_state
        self._executors = {
            PatternType.PARALLEL_SPECIALIST: ParallelSpecialistExecutor(llm_provider),
            PatternType.ROUTER: RouterExecutor(llm_provider),
            PatternType.PLANNER_EXECUTOR_OBSERVER: PlannerExecutorObserverExecutor(llm_provider),
        }

    async def run(self, workflow_id: str, input: str | dict[str, Any]) -> WorkflowResult:
        """Execute a workflow by id and return the final result.

        Accepts either a plain string (existing workflows) or a structured dict
        (e.g. pr_review).  Dict inputs are JSON-serialised to a string for the
        agent pipeline and also stored in SharedState under "workflow_input" so
        downstream components can retrieve the original structure if needed.

        Raises WorkflowNotFound if workflow_id is not registered.
        Re-raises any exception from the pattern executor after marking the run FAILED.
        """
        # Normalise to string for the pipeline; preserve original dict in SharedState.
        if isinstance(input, dict):
            workflow_input_str: str = json.dumps(input)
        else:
            workflow_input_str = input

        wf_def = self._workflow_registry.get(workflow_id)  # type: ignore[attr-defined]
        run = self._run_manager.create_run(workflow_id, workflow_input_str)  # type: ignore[attr-defined]

        if isinstance(input, dict):
            self._shared_state.set(run.run_id, "workflow_input", input)  # type: ignore[attr-defined]

        self._run_manager.update_status(run.run_id, RunStatus.RUNNING)  # type: ignore[attr-defined]
        self._observer.on_event(  # type: ignore[attr-defined]
            WorkflowStartedEvent(
                run_id=run.run_id,
                data={"workflow_id": workflow_id, "input": workflow_input_str},
            )
        )
        context = self._build_context(run.run_id, wf_def)
        executor = self._executors[wf_def.pattern]
        try:
            result = await executor.execute(context, workflow_input_str)
            self._run_manager.complete(run.run_id, result.output)  # type: ignore[attr-defined]
            self._observer.on_event(  # type: ignore[attr-defined]
                WorkflowCompletedEvent(run_id=run.run_id, data={"output": result.output})
            )
            return result
        except Exception as exc:
            self._run_manager.fail(run.run_id, str(exc))  # type: ignore[attr-defined]
            self._observer.on_event(  # type: ignore[attr-defined]
                WorkflowFailedEvent(run_id=run.run_id, data={"error": str(exc)})
            )
            raise

    def _build_context(self, run_id: str, workflow_definition: WorkflowDefinition) -> ExecutionContext:
        return ExecutionContext(
            run_id=run_id,
            workflow_definition=workflow_definition,
            shared_state=self._shared_state,
            workflow_registry=self._workflow_registry,
            agent_registry=self._agent_registry,
            tool_registry=self._tool_registry,
            memory_store=self._memory_store,
            policy_engine=self._policy_engine,
            observer=self._observer,
        )
