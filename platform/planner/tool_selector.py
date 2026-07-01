"""ToolSelector — deterministic tool selection from CapabilityRegistry."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GoalAnalysis, RuntimeAgentDefinition


class ToolSelector:
    """Selects tool names based on the runtime agent team.

    Two-path logic:
    - Static agents (generated=False): looks up required_tool_capabilities via
      registry and finds the tools that satisfy them.
    - Generated agents (generated=True): uses tool_names already assigned by
      RuntimeAgentGenerator (tools mapped from capability tags at generation time).

    Deduplicates by tool_name while preserving encounter order.
    """

    def select(
        self,
        analysis: GoalAnalysis,
        runtime_agents: list[RuntimeAgentDefinition],
        registry: CapabilityRegistry,
    ) -> list[str]:
        """Return ordered list of tool names needed by the full runtime agent team."""
        if not runtime_agents:
            return []

        tool_names: list[str] = []
        seen: set[str] = set()

        for agent in runtime_agents:
            if not agent.generated:
                # Static path: look up required_tool_capabilities in registry
                agent_desc = registry.get_agent(agent.id)
                if agent_desc is None:
                    continue
                for req_cap in agent_desc.required_tool_capabilities:
                    for tool_desc in registry.find_tools_by_capability(req_cap):
                        if tool_desc.tool_name not in seen:
                            seen.add(tool_desc.tool_name)
                            tool_names.append(tool_desc.tool_name)
            else:
                # Generated path: tool_names already assigned by RuntimeAgentGenerator
                for tool_name in agent.tool_names:
                    if tool_name not in seen:
                        seen.add(tool_name)
                        tool_names.append(tool_name)

        return tool_names
