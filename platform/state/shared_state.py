"""SharedState — per-run key-value store for cross-agent data sharing."""

from __future__ import annotations

from typing import Any


class SharedState:
    """In-memory key-value store scoped to a single workflow run.

    Pattern executors write intermediate outputs here so downstream agents
    can read them without passing all data through prompt strings alone.
    Example: planner writes 'plan' → executor reads 'plan'.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Store a value under the given key."""
        # TODO: implement
        raise NotImplementedError

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for key, or default if not set."""
        # TODO: implement
        raise NotImplementedError

    def get_all(self) -> dict[str, Any]:
        """Return a shallow copy of all stored key-value pairs."""
        # TODO: implement
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all stored values."""
        # TODO: implement
        raise NotImplementedError
