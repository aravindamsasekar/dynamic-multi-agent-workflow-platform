"""GoalAnalyzer — the single LLM planning step in V3.2."""

from __future__ import annotations

import json
import re
from typing import Any

from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.message import Message, Role, TextContent
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GoalAnalysis,
    PlannerError,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a workflow planning assistant for a multi-agent platform.

Analyze the user's goal and return a single JSON object. No explanation, no markdown, \
no text outside the JSON.

## Your task

Read the user goal and identify ONLY which capabilities (from the allow-list below) are \
needed to complete it. Do not classify the goal type. Do not choose patterns, agents, or \
tools. Do not invent capability names.

## Capability allow-list

The ONLY valid capability names you may use in required_capabilities:

{capability_list}

## Available registry summary

{registry_summary}

## JSON schema

{{
  "required_capabilities": ["capability_name", ...],
  "risk_level": "low" or "medium" or "high" or "critical",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence>",
  "constraints": ["constraint_name", ...],
  "requires_hitl": true or false
}}

## Rules

- required_capabilities must ONLY contain names from the capability allow-list above. \
Never invent capability names.
- Return required_capabilities = [] when no listed capabilities match the goal.
- Set confidence = 0.0 when no capabilities match.
- PR reviews are read-only by default: risk_level = "low", constraints = ["read_only"].
- Set risk_level = "high" if the goal mentions production, sensitive data, or \
financial systems.
- Set requires_hitl = true when risk_level is "high" or "critical".
- Set requires_hitl = true if the goal contains any of: \
deploy, delete, send, post, notify, refund, production.

Return ONLY the JSON object.\
"""

# ---------------------------------------------------------------------------
# Required fields in LLM response
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "required_capabilities",
    "risk_level",
    "confidence",
    "reasoning",
    "constraints",
    "requires_hitl",
})

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_text(content: list[Any]) -> str:
    for block in content:
        if isinstance(block, TextContent):
            return block.text
    return ""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_goal_analysis(raw: str, allowed_capabilities: frozenset[str]) -> GoalAnalysis:
    """Parse LLM text into a GoalAnalysis. Raises PlannerError on any failure."""
    text = _strip_code_fences(raw)

    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"LLM returned invalid JSON: {exc}") from exc

    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise PlannerError(
            f"LLM response missing required fields: {sorted(missing)}"
        )

    try:
        risk_level = RiskLevel(data["risk_level"])
    except ValueError as exc:
        raise PlannerError(f"Invalid risk_level: {data['risk_level']!r}") from exc

    raw_confidence = data["confidence"]
    if not isinstance(raw_confidence, (int, float)):
        raise PlannerError(
            f"confidence must be a number, got {type(raw_confidence).__name__}"
        )
    confidence = float(raw_confidence)
    if not 0.0 <= confidence <= 1.0:
        raise PlannerError(
            f"confidence must be between 0.0 and 1.0, got {confidence}"
        )

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for cap in data["required_capabilities"]:
        if cap not in seen:
            seen.add(cap)
            deduped.append(cap)

    # Anti-hallucination filter: unknown caps → missing_capabilities
    valid_caps = [cap for cap in deduped if cap in allowed_capabilities]
    missing_caps = [cap for cap in deduped if cap not in allowed_capabilities]

    return GoalAnalysis(
        required_capabilities=valid_caps,
        risk_level=risk_level,
        confidence=confidence,
        reasoning=str(data["reasoning"]),
        constraints=list(data["constraints"]),
        requires_hitl=bool(data["requires_hitl"]),
        missing_capabilities=missing_caps,
    )


# ---------------------------------------------------------------------------
# GoalAnalyzer
# ---------------------------------------------------------------------------


class GoalAnalyzer:
    """Converts a natural-language goal into a structured GoalAnalysis.

    Makes exactly one LLM call per analyze() invocation. All downstream
    planner steps are deterministic and receive the GoalAnalysis as input.

    Args:
        llm:      Any ILLMProvider. Tests inject MockLLMProvider.
        registry: CapabilityRegistry whose summary and capability allow-list
                  are injected into the system prompt.
    """

    def __init__(self, llm: ILLMProvider, registry: CapabilityRegistry) -> None:
        self._llm = llm
        self._registry = registry

    def _build_system_prompt(self) -> str:
        allowed = sorted(self._registry.all_agent_capabilities())
        capability_list = "\n".join(f"- {cap}" for cap in allowed)
        return _SYSTEM_PROMPT_TEMPLATE.format(
            capability_list=capability_list,
            registry_summary=self._registry.to_prompt_summary(),
        )

    async def analyze(self, goal: str) -> GoalAnalysis:
        """Analyze a user goal and return a GoalAnalysis.

        Raises:
            PlannerError: If the LLM returns unparseable or invalid output.
        """
        messages = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt()),
            Message(role=Role.USER, content=goal),
        ]
        response = await self._llm.complete(messages)

        raw_text = _extract_text(response.content)
        if not raw_text:
            raise PlannerError("LLM returned no text content")

        allowed = frozenset(self._registry.all_agent_capabilities())
        return _parse_goal_analysis(raw_text, allowed)
