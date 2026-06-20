"""ExecutionContext — single context object passed to all pattern executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platform.core.models.workflow import WorkflowDefinition


@dataclass
class ExecutionContext:
    """Bundles all runtime dependencies for a workflow execution.

    Passed by the Orchestrator to every PatternExecutor so each pattern
    receives one clean object instead of many individual parameters.

    Concrete types for Any-typed fields:
        shared_state      -> platform.state.shared_state.SharedState
        workflow_registry -> platform.registries.workflow_registry.WorkflowRegistry
        agent_registry    -> platform.registries.agent_registry.AgentRegistry
        tool_registry     -> platform.registries.tool_registry.ToolRegistry
        memory_store      -> platform.memory.in_memory_store.InMemoryStore (IMemoryStore)
        policy_engine     -> platform.policy.engine.PolicyEngine (IPolicyEngine)
        observer          -> platform.observability.console_observer.ConsoleObserver (IObserver)
    """

    run_id: str
    workflow_definition: WorkflowDefinition
    shared_state: Any
    workflow_registry: Any
    agent_registry: Any
    tool_registry: Any
    memory_store: Any
    policy_engine: Any
    observer: Any
