"""ContentFilterRule — blocks requests containing prohibited content."""

from __future__ import annotations

from typing import Any

from platform.core.interfaces.policy import IRule, PolicyDecision


class ContentFilterRule(IRule):
    """Checks message content against a keyword blocklist.

    Case-insensitive. Reads the "content" key from the evaluation context.
    Returns BLOCK if any blocked term is found, ALLOW otherwise.

    Context convention (set by AgentRuntime):
        {"hook": "pre_agent", "agent_id": "...", "content": "<text to inspect>"}
    """

    def __init__(self, blocked_terms: list[str] | None = None) -> None:
        self._blocked_terms: list[str] = [t.lower() for t in (blocked_terms or [])]

    def check(self, context: dict[str, Any]) -> PolicyDecision:
        """Return BLOCK if any blocked term appears in context["content"], else ALLOW."""
        if not self._blocked_terms:
            return PolicyDecision.ALLOW
        content = str(context.get("content", "")).lower()
        for term in self._blocked_terms:
            if term in content:
                return PolicyDecision.BLOCK
        return PolicyDecision.ALLOW
