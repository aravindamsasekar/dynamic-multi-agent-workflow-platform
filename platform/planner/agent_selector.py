"""AgentSelector — deterministic agent selection from CapabilityRegistry."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GoalAnalysis, TaskType


class AgentSelector:
    """Selects agent IDs from the registry based on GoalAnalysis.

    V3.1: returns all agents whose supported_task_types include the goal's
    task_type. No agent generation; every returned ID maps to an existing
    registered agent.
    """

    def select(self, analysis: GoalAnalysis, registry: CapabilityRegistry) -> list[str]:
        """Return ordered list of agent IDs to include in the generated plan."""
        if analysis.task_type == TaskType.UNSUPPORTED:
            return []
        agents = registry.find_agents_by_task_type(analysis.task_type.value)
        return [a.agent_id for a in agents]
