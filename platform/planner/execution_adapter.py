"""ExecutionAdapter — converts GeneratedWorkflowPlan to WorkflowDefinition and runs it."""

from __future__ import annotations

from typing import Any

from platform.core.models.agent import AgentDefinition, LLMConfig
from platform.core.models.workflow import PatternType, WorkflowDefinition, WorkflowResult
from platform.orchestrator.orchestrator import Orchestrator
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GeneratedWorkflowPlan, RuntimeAgentDefinition
from platform.registries.agent_registry import AgentRegistry
from platform.registries.workflow_registry import WorkflowRegistry


class ExecutionAdapter:
    """Bridges the planner and the V2 execution engine.

    Converts a GeneratedWorkflowPlan to a WorkflowDefinition, temporarily registers
    generated agents in AgentRegistry, registers the workflow in WorkflowRegistry,
    and calls Orchestrator.run(). Generated agent registrations are always cleaned up
    via try/finally — this class is the sole lifecycle owner of generated agent entries.

    Ownership rule: no other component may register or unregister generated agents.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        workflow_registry: WorkflowRegistry,
        agent_registry: AgentRegistry,
        capability_registry: CapabilityRegistry,
    ) -> None:
        self._orchestrator = orchestrator
        self._workflow_registry = workflow_registry
        self._agent_registry = agent_registry
        self._capability_registry = capability_registry

    @staticmethod
    def _to_agent_definition(agent: RuntimeAgentDefinition) -> AgentDefinition:
        """Convert a generated RuntimeAgentDefinition to an AgentDefinition.

        All runtime concepts (llm_config, memory, guardrails, output schema)
        default to platform defaults — identical to static agents loaded from YAML.
        This makes generated agents indistinguishable from static agents at runtime.
        """
        return AgentDefinition(
            agent_id=agent.id,
            name=agent.name,
            description=agent.description,
            system_prompt=agent.system_prompt,
            tool_names=agent.tool_names,
            llm_config=LLMConfig(),
        )

    def _find_reviewer_id(self, plan: GeneratedWorkflowPlan) -> str | None:
        """Return the agent_id of the synthesis/reviewer agent, or None.

        Only agents with an explicit 'synthesize_findings' capability are eligible
        as reviewers. No fallback to the last agent — an unrelated agent running as
        reviewer produces wrong output because its system prompt and tools are not
        designed for synthesis.

        When runtime_agents is populated, reads capabilities directly.
        Falls back to selected_agents + capability_registry for old plans.
        """
        if plan.runtime_agents:
            for agent in plan.runtime_agents:
                if "synthesize_findings" in agent.capabilities:
                    return agent.id
            return None  # No synthesis-capable agent — do not add reviewer
        # Backward compat: runtime_agents empty → use selected_agents + registry lookup
        for agent_id in plan.selected_agents:
            desc = self._capability_registry.get_agent(agent_id)
            if desc and "synthesize_findings" in desc.capabilities:
                return agent_id
        return None  # No synthesizer found — do not add reviewer

    def to_workflow_definition(self, plan: GeneratedWorkflowPlan) -> WorkflowDefinition:
        """Build a WorkflowDefinition from a GeneratedWorkflowPlan.

        Uses runtime_agents for agent_ids when populated (covers both static and
        generated agents). Falls back to selected_agents for old plans where
        runtime_agents was not serialized.
        """
        pattern = PatternType(plan.selected_pattern)

        agent_ids = (
            [r.id for r in plan.runtime_agents]
            if plan.runtime_agents
            else plan.selected_agents
        )

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
            agent_ids=agent_ids,
            pattern_config=pattern_config,
            hitl_enabled=plan.hitl_required,
        )

    async def execute(
        self,
        plan: GeneratedWorkflowPlan,
        input_data: str | dict[str, Any],
    ) -> WorkflowResult:
        """Register the plan as a WorkflowDefinition and execute it via the orchestrator.

        Generated agents are temporarily registered in AgentRegistry so the orchestrator
        can resolve them by ID. Cleanup is guaranteed via try/finally regardless of
        whether execution succeeds or raises.
        """
        # When no explicit input is provided, use the plan's user goal as the task.
        # This gives generated agents the original natural-language intent so the LLM
        # can map it to tool arguments (e.g. "Read README.md" → path="README.md").
        # Callers that supply real input (e.g. PR review structured payload) are unaffected.
        effective_input: str | dict = input_data if input_data else plan.user_goal

        registered_ids: list[str] = []
        try:
            for agent in plan.runtime_agents:
                if agent.generated and not self._agent_registry.exists(agent.id):
                    self._agent_registry.register(self._to_agent_definition(agent))
                    registered_ids.append(agent.id)
            wf_def = self.to_workflow_definition(plan)
            self._workflow_registry.register(wf_def)
            return await self._orchestrator.run(plan.plan_id, effective_input)
        finally:
            for agent_id in registered_ids:
                self._agent_registry.unregister(agent_id)
