"""PlannerService — single entry point for plan generation."""

from __future__ import annotations

from platform.core.interfaces.llm import ILLMProvider
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.goal_analyzer import GoalAnalyzer
from platform.planner.models import GeneratedWorkflowPlan, ValidationResult
from platform.planner.plan_builder import PlanBuilder
from platform.planner.plan_validator import PlanValidator
from platform.planner.task_label_inferer import TaskLabelInferer


class PlannerService:
    """Ties GoalAnalyzer → PlanBuilder → TaskLabelInferer → PlanValidator into one call.

    The single entry point callers (API layer) use to go from a user goal
    to a GeneratedWorkflowPlan with a ValidationResult.

    task_label is inferred from the plan's required_capabilities after building
    and before validation, so it is always set on the returned plan.
    """

    def __init__(self, llm: ILLMProvider, registry: CapabilityRegistry) -> None:
        self._analyzer = GoalAnalyzer(llm=llm, registry=registry)
        self._builder = PlanBuilder(registry=registry)
        self._validator = PlanValidator()
        self._label_inferer = TaskLabelInferer()
        self._registry = registry

    async def generate(self, goal: str) -> tuple[GeneratedWorkflowPlan, ValidationResult]:
        """Analyze the goal, build a plan, infer its label, validate, and return both.

        Raises:
            PlannerError: If the LLM returns an unparseable or invalid response.
        """
        analysis = await self._analyzer.analyze(goal)
        plan = self._builder.build(goal, analysis)
        plan.task_label = self._label_inferer.infer(plan.goal_analysis.required_capabilities)
        validation = self._validator.validate(plan, self._registry)
        return plan, validation
