"""Parallel Specialist pattern executor."""

from __future__ import annotations

import asyncio

from platform.aggregator.result_aggregator import AggregationStrategy, ResultAggregator
from platform.agent.runtime import AgentRuntime
from platform.core.exceptions import PatternExecutionError
from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.agent import AgentResult
from platform.core.models.context import ExecutionContext
from platform.core.models.workflow import WorkflowResult
from platform.patterns.base import IPatternExecutor


class ParallelSpecialistExecutor(IPatternExecutor):
    """Runs specialist agents concurrently, aggregates results, optional reviewer pass.

    Pattern flow:
    1. Resolve specialist agent definitions from context.agent_registry
    2. Run all specialists in parallel via asyncio.gather using AgentRuntime
    3. Collect list[AgentResult]
    4. Pass results to ResultAggregator with the configured strategy
    5. If pattern_config defines a reviewer_agent_id, run reviewer via AgentRuntime
    6. Emit WorkflowCompletedEvent and return WorkflowResult

    pattern_config keys:
        strategy          (str, default "concatenate") — AggregationStrategy value
        reviewer_agent_id (str, optional)              — runs after aggregation
    """

    def __init__(self, llm_provider: ILLMProvider) -> None:
        self._llm_provider = llm_provider

    async def execute(self, context: ExecutionContext, workflow_input: str) -> WorkflowResult:
        # TODO Phase 7: integrate with RunManager lifecycle (status transitions, failure propagation)
        wf = context.workflow_definition
        agent_defs = [context.agent_registry.get(aid) for aid in wf.agent_ids]

        coros = [
            AgentRuntime(self._llm_provider, context).run(agent_def, workflow_input)
            for agent_def in agent_defs
        ]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        errors = [
            (agent_defs[i].agent_id, e)
            for i, e in enumerate(outcomes)
            if isinstance(e, BaseException)
        ]
        if errors:
            agent_id, exc = errors[0]
            raise PatternExecutionError(
                f"Agent '{agent_id}' failed: {exc}"
            ) from exc

        results: list[AgentResult] = list(outcomes)  # type: ignore[arg-type]

        raw_strategy = wf.pattern_config.get("strategy", "concatenate")
        strategy = AggregationStrategy(raw_strategy)
        output = ResultAggregator().aggregate(results, strategy)

        reviewer_id: str | None = wf.pattern_config.get("reviewer_agent_id")
        if reviewer_id:
            reviewer_def = context.agent_registry.get(reviewer_id)
            reviewer_result = await AgentRuntime(self._llm_provider, context).run(
                reviewer_def, output
            )
            results.append(reviewer_result)
            output = reviewer_result.output

        return WorkflowResult(
            run_id=context.run_id,
            workflow_id=wf.workflow_id,
            output=output,
            agent_results=results,
        )
