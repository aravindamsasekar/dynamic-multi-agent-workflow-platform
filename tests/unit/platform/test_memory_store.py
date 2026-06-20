"""Unit tests for InMemoryStore."""

from __future__ import annotations

import pytest

from platform.core.models.message import Message, Role
from platform.memory.in_memory_store import InMemoryStore


def make_message(text: str, role: Role = Role.USER) -> Message:
    return Message(role=role, content=text)


class TestInMemoryStore:
    def test_append_and_get_history(self):
        store = InMemoryStore()
        m1 = make_message("hello")
        m2 = make_message("world", Role.ASSISTANT)
        store.append("run-1", "agent-1", m1)
        store.append("run-1", "agent-1", m2)
        history = store.get_history("run-1", "agent-1")
        assert history == [m1, m2]

    def test_get_history_preserves_order(self):
        store = InMemoryStore()
        messages = [make_message(f"msg-{i}") for i in range(5)]
        for m in messages:
            store.append("run-1", "agent-1", m)
        assert store.get_history("run-1", "agent-1") == messages

    def test_get_history_returns_empty_list_for_unknown_pair(self):
        store = InMemoryStore()
        assert store.get_history("run-x", "agent-x") == []

    def test_get_history_returns_copy(self):
        store = InMemoryStore()
        store.append("run-1", "agent-1", make_message("hi"))
        history = store.get_history("run-1", "agent-1")
        history.append(make_message("injected"))
        # original must be unaffected
        assert len(store.get_history("run-1", "agent-1")) == 1

    def test_clear_removes_specific_pair(self):
        store = InMemoryStore()
        store.append("run-1", "agent-1", make_message("hi"))
        store.append("run-1", "agent-2", make_message("there"))
        store.clear("run-1", "agent-1")
        assert store.get_history("run-1", "agent-1") == []
        assert len(store.get_history("run-1", "agent-2")) == 1

    def test_clear_on_nonexistent_pair_is_safe(self):
        store = InMemoryStore()
        store.clear("run-x", "agent-x")  # must not raise

    def test_clear_run_removes_all_agents_for_run(self):
        store = InMemoryStore()
        store.append("run-1", "agent-1", make_message("a"))
        store.append("run-1", "agent-2", make_message("b"))
        store.append("run-2", "agent-1", make_message("c"))
        store.clear_run("run-1")
        assert store.get_history("run-1", "agent-1") == []
        assert store.get_history("run-1", "agent-2") == []
        # run-2 must be unaffected
        assert len(store.get_history("run-2", "agent-1")) == 1

    def test_clear_run_on_nonexistent_run_is_safe(self):
        store = InMemoryStore()
        store.clear_run("nonexistent")  # must not raise

    def test_isolation_between_runs(self):
        store = InMemoryStore()
        store.append("run-A", "agent-1", make_message("from A"))
        store.append("run-B", "agent-1", make_message("from B"))
        assert store.get_history("run-A", "agent-1")[0].content == "from A"
        assert store.get_history("run-B", "agent-1")[0].content == "from B"

    def test_isolation_between_agents_same_run(self):
        store = InMemoryStore()
        store.append("run-1", "agent-1", make_message("for agent 1"))
        store.append("run-1", "agent-2", make_message("for agent 2"))
        assert store.get_history("run-1", "agent-1")[0].content == "for agent 1"
        assert store.get_history("run-1", "agent-2")[0].content == "for agent 2"
