"""IPolicyEngine and IRule interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


class IRule(ABC):
    """Abstract interface for a single policy rule."""

    @abstractmethod
    def check(self, context: dict[str, Any]) -> PolicyDecision:
        """Evaluate this rule against the provided context."""
        ...


class IPolicyEngine(ABC):
    """Abstract interface for the policy engine."""

    @abstractmethod
    def evaluate(self, hook: str, context: dict[str, Any]) -> None:
        """Evaluate all rules at the given hook point. Raises PolicyViolation if blocked."""
        ...
