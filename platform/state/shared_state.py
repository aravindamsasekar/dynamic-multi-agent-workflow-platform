"""SharedState — per-run key-value store for cross-agent data sharing."""

from __future__ import annotations

from typing import Any


class SharedState:
    """In-memory key-value store scoped by run_id.

    Pattern executors write intermediate outputs here so downstream agents
    can read them without passing all data through prompt strings alone.
    Example: planner writes 'plan' → executor reads 'plan'.

    Each run_id gets its own isolated bucket; operations on one run never
    affect another.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def set(self, run_id: str, key: str, value: Any) -> None:
        """Store a value under key for the given run."""
        if run_id not in self._data:
            self._data[run_id] = {}
        self._data[run_id][key] = value

    def get(self, run_id: str, key: str, default: Any = None) -> Any:
        """Return the value for key in the given run, or default if not set."""
        return self._data.get(run_id, {}).get(key, default)

    def update(self, run_id: str, data: dict[str, Any]) -> None:
        """Merge all key-value pairs from data into the given run's bucket."""
        if run_id not in self._data:
            self._data[run_id] = {}
        self._data[run_id].update(data)

    def get_all(self, run_id: str) -> dict[str, Any]:
        """Return a shallow copy of all key-value pairs for the given run."""
        return dict(self._data.get(run_id, {}))

    def clear(self, run_id: str) -> None:
        """Remove all stored values for the given run."""
        self._data.pop(run_id, None)
