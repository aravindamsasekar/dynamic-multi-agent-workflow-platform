"""AgentSelector — deterministic, capability-first agent selection.

Deprecated since Phase B: PlanBuilder now uses RuntimeAgentGenerator, which
returns RuntimeAgentDefinition objects covering both static and generated agents.
AgentSelector is retained for reference but is no longer called by the platform.
"""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GoalAnalysis


class AgentSelector:
    """Selects agent IDs from the registry based on GoalAnalysis.

    V3.2: capability-first. For each required capability, finds all agents
    that cover it. Returns a deduplicated list preserving first-encounter order
    so agent ordering is stable and deterministic.
    """

    def select(self, analysis: GoalAnalysis, registry: CapabilityRegistry) -> list[str]:
        """Return an ordered list of agent IDs to include in the generated plan."""
        if not analysis.required_capabilities:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for cap in analysis.required_capabilities:
            for agent in registry.find_agents_by_capability(cap):
                if agent.agent_id not in seen:
                    seen.add(agent.agent_id)
                    result.append(agent.agent_id)
        return result
