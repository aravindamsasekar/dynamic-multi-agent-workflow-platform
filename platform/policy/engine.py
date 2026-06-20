"""PolicyEngine — evaluates rules at agent and tool hook points."""

from __future__ import annotations

from enum import Enum
from typing import Any

from platform.core.interfaces.policy import IPolicyEngine, IRule


class HookPoint(str, Enum):
    PRE_AGENT = "pre_agent"
    POST_AGENT = "post_agent"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"


class PolicyEngine(IPolicyEngine):
    """Evaluates registered IRule instances at defined hook points.

    Called by AgentRuntime at PRE_AGENT, POST_AGENT, PRE_TOOL, POST_TOOL.
    Raises PolicyViolation if any rule returns BLOCK.
    Emits PolicyViolationEvent to observer on WARN decisions.
    """

    def __init__(self, rules: list[IRule] | None = None) -> None:
        self._rules: list[IRule] = rules or []

    def add_rule(self, rule: IRule) -> None:
        """Register an additional rule with this engine."""
        # TODO: implement
        raise NotImplementedError

    def evaluate(self, hook: str, context: dict[str, Any]) -> None:
        """Evaluate all rules at the given hook point. Raises PolicyViolation if blocked."""
        # TODO: implement
        raise NotImplementedError
