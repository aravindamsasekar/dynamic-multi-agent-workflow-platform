"""PolicyEngine — evaluates rules at agent and tool hook points."""

from __future__ import annotations

from enum import Enum
from typing import Any

from platform.core.exceptions import PolicyViolation
from platform.core.interfaces.policy import IPolicyEngine, IRule, PolicyDecision


class HookPoint(str, Enum):
    PRE_AGENT = "pre_agent"
    POST_AGENT = "post_agent"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"


class PolicyEngine(IPolicyEngine):
    """Evaluates registered IRule instances at defined hook points.

    Called by AgentRuntime at PRE_AGENT, POST_AGENT, PRE_TOOL, POST_TOOL.
    Raises PolicyViolation if any rule returns BLOCK.
    WARN decisions pass silently in V1 (observer not wired here).
    """

    def __init__(self, rules: list[IRule] | None = None) -> None:
        self._rules: list[IRule] = rules or []

    def add_rule(self, rule: IRule) -> None:
        """Register an additional rule with this engine."""
        self._rules.append(rule)

    def evaluate(self, hook: str, context: dict[str, Any]) -> None:
        """Evaluate all rules at the given hook point. Raises PolicyViolation if blocked."""
        for rule in self._rules:
            decision = rule.check(context)
            if decision == PolicyDecision.BLOCK:
                hook_name = hook.value if hasattr(hook, "value") else hook
                raise PolicyViolation(
                    f"Policy blocked execution at hook '{hook_name}': "
                    f"{rule.__class__.__name__}"
                )
