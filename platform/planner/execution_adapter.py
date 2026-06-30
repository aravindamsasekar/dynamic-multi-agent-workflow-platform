"""ExecutionAdapter — converts GeneratedWorkflowPlan to WorkflowDefinition and runs it."""

from __future__ import annotations

from typing import Any

from platform.core.models.workflow import PatternType, WorkflowDefinition, WorkflowResult
from platform.orchestrator.orchestrator import Orchestrator
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GeneratedWorkflowPlan
from platform.registries.workflow_registry import WorkflowRegistry


class ExecutionAdapter:
    """Bridges the planner and the V2 execution engine.

    Converts a GeneratedWorkflowPlan to a WorkflowDefinition, registers it
    temporarily in the WorkflowRegistry, and calls Orchestrator.run().
    No V2 runtime code is modified.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        workflow_registry: WorkflowRegistry,
        capability_registry: CapabilityRegistry,
    ) -> None:
        self._orchestrator = orchestrator
        self._workflow_registry = workflow_registry
        self._capability_registry = capability_registry

    def _find_reviewer_id(self, plan: GeneratedWorkflowPlan) -> str | None:
        """Return the agent_id of the reviewer (synthesis) agent, or None."""
        for agent_id in plan.selected_agents:
            desc = self._capability_registry.get_agent(agent_id)
            if desc and "synthesize_findings" in desc.capabilities:
                return agent_id
        return plan.selected_agents[-1] if plan.selected_agents else None

    def to_workflow_definition(self, plan: GeneratedWorkflowPlan) -> WorkflowDefinition:
        """Build a WorkflowDefinition from a GeneratedWorkflowPlan."""
        pattern = PatternType(plan.selected_pattern)

        pattern_config: dict[str, Any] = {}
        if pattern == PatternType.PARALLEL_SPECIALIST:
            reviewer = self._find_reviewer_id(plan)
            pattern_config = {
                "strategy": "concatenate",
                **({"reviewer_agent_id": reviewer} if reviewer else {}),
            }

        return WorkflowDefinition(
            workflow_id=plan.plan_id,
            name=f"Generated: {plan.user_goal[:60]}",
            description=plan.explanation,
            pattern=pattern,
            agent_ids=plan.selected_agents,
            pattern_config=pattern_config,
            hitl_enabled=plan.hitl_required,
        )

    async def execute(
        self,
        plan: GeneratedWorkflowPlan,
        input_data: str | dict[str, Any],
    ) -> WorkflowResult:
        """Register the plan as a WorkflowDefinition and execute it via the orchestrator."""
        wf_def = self.to_workflow_definition(plan)
        # Register so orchestrator.run(plan_id, ...) can look it up.
        # plan_id is a UUID so there's no collision with static workflow IDs.
        self._workflow_registry.register(wf_def)
        return await self._orchestrator.run(plan.plan_id, input_data)
