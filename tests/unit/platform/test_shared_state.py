"""Unit tests for SharedState."""

from __future__ import annotations

import pytest

from platform.state.shared_state import SharedState


class TestSharedState:
    def test_set_and_get(self):
        state = SharedState()
        state.set("run-1", "plan", "step 1, step 2")
        assert state.get("run-1", "plan") == "step 1, step 2"

    def test_get_returns_default_when_key_absent(self):
        state = SharedState()
        assert state.get("run-1", "missing") is None
        assert state.get("run-1", "missing", "fallback") == "fallback"

    def test_get_returns_default_when_run_absent(self):
        state = SharedState()
        assert state.get("nonexistent-run", "key") is None

    def test_update_merges_dict(self):
        state = SharedState()
        state.set("run-1", "a", 1)
        state.update("run-1", {"b": 2, "c": 3})
        assert state.get("run-1", "a") == 1
        assert state.get("run-1", "b") == 2
        assert state.get("run-1", "c") == 3

    def test_update_overwrites_existing_keys(self):
        state = SharedState()
        state.set("run-1", "key", "old")
        state.update("run-1", {"key": "new"})
        assert state.get("run-1", "key") == "new"

    def test_update_creates_run_bucket_if_absent(self):
        state = SharedState()
        state.update("run-new", {"x": 10})
        assert state.get("run-new", "x") == 10

    def test_get_all_returns_copy(self):
        state = SharedState()
        state.set("run-1", "k", "v")
        result = state.get_all("run-1")
        result["extra"] = "injected"
        # original must be unaffected
        assert state.get("run-1", "extra") is None

    def test_get_all_returns_empty_dict_for_unknown_run(self):
        state = SharedState()
        assert state.get_all("nonexistent") == {}

    def test_clear_removes_run_bucket(self):
        state = SharedState()
        state.set("run-1", "k", "v")
        state.clear("run-1")
        assert state.get("run-1", "k") is None
        assert state.get_all("run-1") == {}

    def test_clear_does_not_affect_other_runs(self):
        state = SharedState()
        state.set("run-1", "k", "v1")
        state.set("run-2", "k", "v2")
        state.clear("run-1")
        assert state.get("run-2", "k") == "v2"

    def test_isolation_between_runs(self):
        state = SharedState()
        state.set("run-A", "shared_key", "A_value")
        state.set("run-B", "shared_key", "B_value")
        assert state.get("run-A", "shared_key") == "A_value"
        assert state.get("run-B", "shared_key") == "B_value"

    def test_set_various_value_types(self):
        state = SharedState()
        state.set("run-1", "int_val", 42)
        state.set("run-1", "list_val", [1, 2, 3])
        state.set("run-1", "dict_val", {"nested": True})
        assert state.get("run-1", "int_val") == 42
        assert state.get("run-1", "list_val") == [1, 2, 3]
        assert state.get("run-1", "dict_val") == {"nested": True}
