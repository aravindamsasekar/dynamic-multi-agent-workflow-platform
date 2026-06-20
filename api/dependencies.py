"""FastAPI dependency injection for platform services."""

from __future__ import annotations

# TODO: initialize platform singletons at startup and expose via dependency functions
# Singletons to wire:
#   WorkflowRegistry, AgentRegistry, ToolRegistry
#   InMemoryStore, SharedState
#   PolicyEngine (with ContentFilterRule)
#   ConsoleObserver
#   RunManager
#   ApprovalManager
#   Orchestrator


def get_orchestrator():
    # TODO: return Orchestrator singleton
    raise NotImplementedError


def get_run_manager():
    # TODO: return RunManager singleton
    raise NotImplementedError


def get_hitl_manager():
    # TODO: return ApprovalManager singleton
    raise NotImplementedError
