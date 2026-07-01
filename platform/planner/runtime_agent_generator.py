"""RuntimeAgentGenerator — deterministic runtime agent generation.

No LLM calls. All agent IDs, names, descriptions, and system prompts are
produced from static templates. Existing static agents are returned as
references (generated=False); capabilities with no static coverage get a
fully-populated generated definition (generated=True).
"""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import RuntimeAgentDefinition


# ---------------------------------------------------------------------------
# Deterministic template helpers
# ---------------------------------------------------------------------------


def _agent_id_for_capability(plan_id: str, capability: str) -> str:
    return f"gen_{plan_id}_{capability}"


def _name_for_capability(capability: str) -> str:
    """'filesystem_read' → 'Filesystem Read Agent'"""
    words = capability.replace("_", " ").split()
    return " ".join(w.capitalize() for w in words) + " Agent"


def _description_for_capability(capability: str) -> str:
    return f"Agent responsible for the '{capability}' capability."


def _system_prompt_for_capability(capability: str) -> str:
    return (
        f"You are responsible for executing the capability '{capability}'. "
        "Use only the assigned tools. Return concise structured results."
    )


def _tools_for_capability(capability: str, registry: CapabilityRegistry) -> list[str]:
    """All tool names that expose `capability`. Deduplicated, stable order."""
    seen: set[str] = set()
    result: list[str] = []
    for tool in registry.find_tools_by_capability(capability):
        if tool.tool_name not in seen:
            seen.add(tool.tool_name)
            result.append(tool.tool_name)
    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class RuntimeAgentGenerator:
    """Produces RuntimeAgentDefinitions for a set of required capabilities.

    For each capability in the input list:
    - If a static registered agent already covers it → return a reference to
      that agent (generated=False, system_prompt=""). The runtime loads the
      full agent definition by ID with no changes to the V2 runtime.
    - If no static agent covers it → generate a new RuntimeAgentDefinition
      from deterministic templates (generated=True). No LLM calls.

    Deduplication rules:
    - Duplicate capabilities in the input are silently collapsed.
    - When two capabilities map to the same static agent only one entry is
      returned (the first encounter wins).
    - Input order is preserved in the output.
    """

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def generate(self, required_capabilities: list[str], plan_id: str) -> list[RuntimeAgentDefinition]:
        """Return one RuntimeAgentDefinition per unique capability/static-agent.

        Args:
            required_capabilities: ordered list of capability tags to fulfil.

        Returns:
            Ordered list of RuntimeAgentDefinitions; empty when input is empty.
        """
        seen_caps: set[str] = set()
        seen_agent_ids: set[str] = set()
        result: list[RuntimeAgentDefinition] = []

        for cap in required_capabilities:
            if cap in seen_caps:
                continue
            seen_caps.add(cap)

            matching = self._registry.find_agents_by_capability(cap)

            if matching:
                static = matching[0]
                if static.agent_id not in seen_agent_ids:
                    seen_agent_ids.add(static.agent_id)
                    result.append(RuntimeAgentDefinition(
                        id=static.agent_id,
                        name=static.name,
                        description=static.description,
                        capabilities=list(static.capabilities),
                        tool_names=[],
                        system_prompt="",
                        generated=False,
                    ))
            else:
                agent_id = _agent_id_for_capability(plan_id, cap)
                if agent_id not in seen_agent_ids:
                    seen_agent_ids.add(agent_id)
                    result.append(RuntimeAgentDefinition(
                        id=agent_id,
                        name=_name_for_capability(cap),
                        description=_description_for_capability(cap),
                        capabilities=[cap],
                        tool_names=_tools_for_capability(cap, self._registry),
                        system_prompt=_system_prompt_for_capability(cap),
                        generated=True,
                    ))

        return result
