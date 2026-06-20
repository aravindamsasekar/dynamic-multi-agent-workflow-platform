"""Planner → Executor → Observer pattern executor."""

from __future__ import annotations

from platform.core.models.context import ExecutionContext
from platform.core.models.workflow import WorkflowResult
from platform.patterns.base import IPatternExecutor


class PlannerExecutorObserverExecutor(IPatternExecutor):
    """Plans steps, executes each, observes result, loops until done or max_iterations.

    Pattern flow:
    1. Run planner agent — produces a structured step list stored in shared_state
    2. For each step (up to pattern_config.max_iterations):
       a. Run executor agent with the current step and shared_state context
       b. Run observer agent to evaluate the executor output
       c. If observer signals DONE → break loop and return WorkflowResult
       d. If observer signals RETRY → continue to next iteration
    3. Return WorkflowResult after all steps complete or max_iterations reached
    """

    async def execute(self, context: ExecutionContext) -> WorkflowResult:
        # TODO: implement
        raise NotImplementedError
