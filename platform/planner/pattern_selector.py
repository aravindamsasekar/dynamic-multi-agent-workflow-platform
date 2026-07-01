"""PatternSelector — deterministic, capability-driven pattern selection."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GoalAnalysis, RuntimeAgentDefinition

# Patterns are evaluated in this priority order.
# The first match (by trigger_capabilities overlap) wins.
# A pattern with empty trigger_capabilities is the universal fallback.
_PRIORITY_ORDER: tuple[str, ...] = (
    "router",
    "planner_executor_observer",
    "parallel_specialist",
)


class PatternSelector:
    """Selects an execution pattern based on the capabilities of runtime agents.

    V3.2: trigger_capabilities-based. Checks patterns in priority order.
    A pattern whose trigger_capabilities overlap with any agent's capabilities
    is chosen. A pattern with empty trigger_capabilities is the universal
    fallback (parallel_specialist).

    Capabilities are read directly from RuntimeAgentDefinition — no registry
    lookup for agent capabilities is performed. The registry is used only for
    pattern descriptor lookups.

    Returns '' if no agents are provided or no registered pattern matches.
    Does not consult task_type.
    """

    def select(
        self,
        analysis: GoalAnalysis,
        runtime_agents: list[RuntimeAgentDefinition],
        registry: CapabilityRegistry,
    ) -> str:
        """Return the pattern name, or '' if no agents are provided or no pattern matches."""
        if not runtime_agents:
            return ""

        agent_caps: set[str] = set()
        for agent in runtime_agents:
            agent_caps.update(agent.capabilities)

        for pattern_name in _PRIORITY_ORDER:
            desc = registry.get_pattern(pattern_name)
            if desc is None:
                continue
            if desc.trigger_capabilities:
                if any(cap in agent_caps for cap in desc.trigger_capabilities):
                    return pattern_name
            else:
                return pattern_name

        return ""
