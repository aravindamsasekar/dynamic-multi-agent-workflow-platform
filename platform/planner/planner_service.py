"""PlannerService — single entry point for plan generation."""

from __future__ import annotations

from platform.core.interfaces.llm import ILLMProvider
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.goal_analyzer import GoalAnalyzer
from platform.planner.models import GeneratedWorkflowPlan, ValidationResult
from platform.planner.plan_builder import PlanBuilder
from platform.planner.plan_validator import PlanValidator


class PlannerService:
    """Ties GoalAnalyzer → PlanBuilder → PlanValidator into one call.

    The single entry point callers (API layer) use to go from a user goal
    to a GeneratedWorkflowPlan with a ValidationResult.
    """

    def __init__(self, llm: ILLMProvider, registry: CapabilityRegistry) -> None:
        self._analyzer = GoalAnalyzer(llm=llm, registry=registry)
        self._builder = PlanBuilder(registry=registry)
        self._validator = PlanValidator()
        self._registry = registry

    async def generate(self, goal: str) -> tuple[GeneratedWorkflowPlan, ValidationResult]:
        """Analyze the goal, build a plan, validate it, and return both.

        Raises:
            PlannerError: If the LLM returns an unparseable or invalid response.
        """
        analysis = await self._analyzer.analyze(goal)
        plan = self._builder.build(goal, analysis)
        validation = self._validator.validate(plan, self._registry)
        return plan, validation
