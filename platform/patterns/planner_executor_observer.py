"""Planner → Executor → Observer pattern executor."""

from __future__ import annotations

from platform.agent.runtime import AgentRuntime
from platform.core.exceptions import PatternExecutionError
from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.agent import AgentResult
from platform.core.models.context import ExecutionContext
from platform.core.models.workflow import WorkflowResult
from platform.patterns.base import IPatternExecutor


class PlannerExecutorObserverExecutor(IPatternExecutor):
    """Plans steps, executes each, observes result, loops until done or max_iterations.

    Pattern flow:
    1. Resolve planner, executor, observer agent definitions from context.agent_registry
    2. Loop up to max_iterations:
       a. Run planner (workflow_input on iter 0; prior result + observer feedback on iter 1+)
       b. Run executor with planner output
       c. Run observer with executor output
       d. If done_signal in observer output → return WorkflowResult(output=executor output)
       e. Else → continue to next iteration
    3. Raise PatternExecutionError if max_iterations exceeded without DONE signal

    pattern_config keys:
        planner_agent_id   (str, required)
        executor_agent_id  (str, required)
        observer_agent_id  (str, required)
        max_iterations     (int, default 5)
        done_signal        (str, default "DONE")
        retry_signal       (str, default "RETRY")
        continue_signal    (str, default "CONTINUE")

    SharedState written each iteration:
        peo_iteration            — current iteration index (0-based)
        peo_last_executor_output — output from the most recent executor call
    """

    def __init__(self, llm_provider: ILLMProvider) -> None:
        self._llm_provider = llm_provider

    async def execute(self, context: ExecutionContext, workflow_input: str) -> WorkflowResult:
        # TODO Phase 7: integrate with RunManager lifecycle (status transitions, failure propagation)
        cfg = context.workflow_definition.pattern_config
        planner_def  = context.agent_registry.get(cfg["planner_agent_id"])
        executor_def = context.agent_registry.get(cfg["executor_agent_id"])
        observer_def = context.agent_registry.get(cfg["observer_agent_id"])
        max_iterations = int(cfg.get("max_iterations", 5))
        done_signal    = cfg.get("done_signal", "DONE").lower()

        last_executor_output = ""
        last_observer_output = ""
        all_agent_results: list[AgentResult] = []

        for i in range(max_iterations):
            if i == 0:
                planner_input = workflow_input
            else:
                planner_input = (
                    f"Iteration {i + 1}.\n"
                    f"Original task: {workflow_input}\n"
                    f"Previous result: {last_executor_output}\n"
                    f"Observer feedback: {last_observer_output}"
                )

            planner_result = await AgentRuntime(self._llm_provider, context).run(
                planner_def, planner_input
            )
            all_agent_results.append(planner_result)

            executor_result = await AgentRuntime(self._llm_provider, context).run(
                executor_def, planner_result.output
            )
            all_agent_results.append(executor_result)
            last_executor_output = executor_result.output

            if context.shared_state is not None:
                context.shared_state.update(context.run_id, {
                    "peo_iteration": i,
                    "peo_last_executor_output": last_executor_output,
                })

            observer_result = await AgentRuntime(self._llm_provider, context).run(
                observer_def, executor_result.output
            )
            all_agent_results.append(observer_result)
            last_observer_output = observer_result.output

            if done_signal in observer_result.output.strip().lower():
                return WorkflowResult(
                    run_id=context.run_id,
                    workflow_id=context.workflow_definition.workflow_id,
                    output=last_executor_output,
                    agent_results=all_agent_results,
                )

        raise PatternExecutionError(
            f"PlannerExecutorObserver exceeded {max_iterations} iterations without a DONE signal"
        )
