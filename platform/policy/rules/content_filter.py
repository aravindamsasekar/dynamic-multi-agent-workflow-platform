"""ContentFilterRule — blocks requests containing prohibited content."""

from __future__ import annotations

from typing import Any

from platform.core.interfaces.policy import IRule, PolicyDecision


class ContentFilterRule(IRule):
    """Checks message content against a keyword blocklist.

    Case-insensitive. Configured with a list of prohibited terms via
    policy_config in workflow.yaml. Returns BLOCK if any term is found.
    """

    def __init__(self, blocked_terms: list[str] | None = None) -> None:
        self._blocked_terms: list[str] = [t.lower() for t in (blocked_terms or [])]

    def check(self, context: dict[str, Any]) -> PolicyDecision:
        # TODO: implement
        raise NotImplementedError
