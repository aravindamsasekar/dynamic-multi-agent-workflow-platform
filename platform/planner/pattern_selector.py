"""PatternSelector — deterministic, capability-driven pattern selection."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GoalAnalysis

# Patterns are evaluated in this priority order.
# The first match (by trigger_capabilities overlap) wins.
# A pattern with empty trigger_capabilities is the universal fallback.
_PRIORITY_ORDER: tuple[str, ...] = (
    "router",
    "planner_executor_observer",
    "parallel_specialist",
)


class PatternSelector:
    """Selects an execution pattern based on the capabilities of selected agents.

    V3.2: trigger_capabilities-based. Checks patterns in priority order.
    A pattern whose trigger_capabilities overlap with any selected agent's
    capabilities is chosen. A pattern with empty trigger_capabilities is the
    universal fallback (parallel_specialist).

    Returns '' if no agents are selected or no registered pattern matches.
    Does not consult task_type.
    """

    def select(
        self,
        analysis: GoalAnalysis,
        selected_agents: list[str],
        registry: CapabilityRegistry,
    ) -> str:
        """Return the pattern name, or '' if no agents are selected or no pattern matches."""
        if not selected_agents:
            return ""

        agent_caps: set[str] = set()
        for agent_id in selected_agents:
            desc = registry.get_agent(agent_id)
            if desc is not None:
                agent_caps.update(desc.capabilities)

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
