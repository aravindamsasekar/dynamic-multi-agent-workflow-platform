"""Router pattern executor."""

from __future__ import annotations

from platform.agent.runtime import AgentRuntime
from platform.core.exceptions import PatternExecutionError
from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.agent import AgentResult
from platform.core.models.context import ExecutionContext
from platform.core.models.workflow import WorkflowResult
from platform.patterns.base import IPatternExecutor


class RouterExecutor(IPatternExecutor):
    """Classifies input and dispatches to the matched specialist agent.

    Pattern flow:
    1. Run classifier agent on the workflow input via AgentRuntime
    2. Parse the route label from classifier output (strip + lowercase)
    3. Look up target agent_id from pattern_config routing table
    4. Run target agent via AgentRuntime with the original workflow input
    5. Return WorkflowResult with target agent output

    pattern_config keys:
        classifier_agent_id (str, required) — agent that outputs the route label
        routes              (dict[str, str]) — label → agent_id routing table
    """

    def __init__(self, llm_provider: ILLMProvider) -> None:
        self._llm_provider = llm_provider

    async def execute(self, context: ExecutionContext, workflow_input: str) -> WorkflowResult:
        # TODO Phase 7: integrate with RunManager lifecycle (status transitions, failure propagation)
        wf = context.workflow_definition
        classifier_agent_id: str = wf.pattern_config["classifier_agent_id"]
        routes: dict[str, str] = wf.pattern_config["routes"]

        classifier_def = context.agent_registry.get(classifier_agent_id)
        classifier_result: AgentResult = await AgentRuntime(
            self._llm_provider, context
        ).run(classifier_def, workflow_input)

        route_label = classifier_result.output.strip().lower()
        target_agent_id = routes.get(route_label)
        if target_agent_id is None:
            raise PatternExecutionError(f"Unknown route: '{route_label}'")

        target_def = context.agent_registry.get(target_agent_id)
        target_result: AgentResult = await AgentRuntime(
            self._llm_provider, context
        ).run(target_def, workflow_input)

        return WorkflowResult(
            run_id=context.run_id,
            workflow_id=wf.workflow_id,
            output=target_result.output,
            agent_results=[classifier_result, target_result],
        )
