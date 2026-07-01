"""PlannerService — single entry point for plan generation."""

from __future__ import annotations

import uuid

from platform.core.interfaces.llm import ILLMProvider
from platform.extensions.manager import CapabilityManager
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.goal_analyzer import GoalAnalyzer
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    PlannerError,
    ValidationError,
    ValidationResult,
)
from platform.planner.plan_builder import PlanBuilder
from platform.planner.plan_validator import PlanValidator
from platform.planner.task_label_inferer import TaskLabelInferer


class PlannerService:
    """Ties GoalAnalyzer → CapabilityManager → PlanBuilder → TaskLabelInferer → PlanValidator.

    Phase 3: after GoalAnalyzer returns required_capabilities, CapabilityManager.resolve()
    is called. If any capability is missing, generate() returns a non-executable pending_install
    plan with install suggestions instead of building a full plan.

    When all capabilities are satisfied, the existing V3 flow is unchanged.
    """

    def __init__(
        self,
        llm: ILLMProvider,
        registry: CapabilityRegistry,
        capability_manager: CapabilityManager | None = None,
    ) -> None:
        # When a CapabilityManager is wired in, expose marketplace catalog capabilities
        # (names + descriptions) to the GoalAnalyzer so the LLM can request them before
        # installation. Without this, the pending_install flow is unreachable because:
        # 1. Uninstalled capabilities are absent from the LLM allow-list, so the LLM
        #    never requests them.
        # 2. Even if the name were present, the LLM has no description to match the goal.
        catalog_caps: frozenset[str] = frozenset()
        catalog_descs: dict[str, str] = {}
        if capability_manager is not None:
            catalog_caps = frozenset(capability_manager.catalog_capabilities())
            catalog_descs = capability_manager.catalog_capability_descriptions()
        self._analyzer = GoalAnalyzer(
            llm=llm,
            registry=registry,
            extra_capabilities=catalog_caps,
            extra_capability_descriptions=catalog_descs,
        )
        self._builder = PlanBuilder(registry=registry)
        self._validator = PlanValidator()
        self._label_inferer = TaskLabelInferer()
        self._registry = registry
        self._capability_manager = capability_manager

    async def generate(self, goal: str) -> tuple[GeneratedWorkflowPlan, ValidationResult]:
        """Analyze the goal, build a plan, infer its label, validate, and return both.

        If CapabilityManager is configured and detects missing capabilities, returns
        a pending_install plan stub with install suggestions instead of a full plan.

        Raises:
            PlannerError: If the LLM returns an unparseable or invalid response.
        """
        analysis = await self._analyzer.analyze(goal)

        if self._capability_manager is not None:
            resolution = self._capability_manager.resolve(analysis.required_capabilities)
            if not resolution.all_satisfied:
                return self._make_pending_install_plan(goal, analysis, resolution)

        plan = self._builder.build(goal, analysis)
        plan.task_label = self._label_inferer.infer(plan.goal_analysis.required_capabilities)
        validation = self._validator.validate(plan, self._registry)
        plan.executable = validation.is_valid
        return plan, validation

    def _make_pending_install_plan(
        self,
        goal: str,
        analysis: GoalAnalysis,
        resolution,
    ) -> tuple[GeneratedWorkflowPlan, ValidationResult]:
        """Return a non-executable plan stub when required capabilities are unavailable."""
        has_unsupported = len(resolution.unsupported) > 0
        if has_unsupported:
            explanation = (
                f"Goal requires capabilities that no marketplace extension can provide: "
                f"{resolution.unsupported}. "
                f"The platform cannot currently fulfill this goal."
            )
        else:
            explanation = (
                f"Goal requires capabilities not yet installed: {resolution.missing}. "
                f"Install the suggested extension(s) to enable this goal."
            )

        plan = GeneratedWorkflowPlan(
            plan_id=str(uuid.uuid4()),
            user_goal=goal,
            goal_analysis=analysis,
            selected_pattern="",
            selected_agents=[],
            selected_tools=[],
            guardrails=[],
            hitl_required=False,
            warnings=[],
            explanation=explanation,
            estimated_complexity="unknown",
            estimated_duration_seconds=0,
            task_label="",
            executable=False,
            missing_capabilities=resolution.missing,
            install_suggestions=resolution.suggestions,
            unsupported=has_unsupported,
        )
        validation = ValidationResult(
            is_valid=False,
            errors=[
                ValidationError(
                    code="MISSING_CAPABILITIES",
                    message=f"Missing capabilities: {resolution.missing}",
                )
            ],
            warnings=[],
        )
        return plan, validation
