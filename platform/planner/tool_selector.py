"""ToolSelector — deterministic tool selection from CapabilityRegistry."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GoalAnalysis


class ToolSelector:
    """Selects tool names from the registry based on the selected agents.

    For each selected agent, looks up its required_tool_capabilities in the
    registry and finds the tools that satisfy them. Deduplicates by tool_name
    while preserving encounter order.
    """

    def select(
        self,
        analysis: GoalAnalysis,
        selected_agents: list[str],
        registry: CapabilityRegistry,
    ) -> list[str]:
        """Return ordered list of tool names needed by the selected agents."""
        if not selected_agents:
            return []

        tool_names: list[str] = []
        seen: set[str] = set()

        for agent_id in selected_agents:
            agent = registry.get_agent(agent_id)
            if agent is None:
                continue
            for req_cap in agent.required_tool_capabilities:
                for tool_desc in registry.find_tools_by_capability(req_cap):
                    if tool_desc.tool_name not in seen:
                        seen.add(tool_desc.tool_name)
                        tool_names.append(tool_desc.tool_name)

        return tool_names
