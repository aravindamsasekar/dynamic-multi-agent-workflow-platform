"""Integration tests for demo workflow configurations.

Each test class loads a real workflow directory via ConfigLoader.load_one(),
builds an ExecutionContext from the populated registries, runs the appropriate
PatternExecutor with a pre-queued MockLLMProvider, and asserts on
WorkflowResult structure and content.

No real LLM API calls are made in any test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from platform.config.loader import ConfigLoader
from platform.core.exceptions import PatternExecutionError
from platform.core.interfaces.observer import IObserver
from platform.core.models.context import ExecutionContext
from platform.core.models.events import WorkflowEvent
from platform.core.models.message import (
    LLMResponse,
    StopReason,
    TextContent,
    ToolUseContent,
)
from platform.core.models.workflow import WorkflowResult
from platform.memory.in_memory_store import InMemoryStore
from platform.patterns.parallel_specialist import ParallelSpecialistExecutor
from platform.patterns.planner_executor_observer import PlannerExecutorObserverExecutor
from platform.patterns.router import RouterExecutor
from platform.policy.engine import PolicyEngine
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry
from platform.llm.mock_provider import MockLLMProvider

_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent.parent / "workflows"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _CapturingObserver(IObserver):
    def __init__(self) -> None:
        self.events: list[WorkflowEvent] = []

    def on_event(self, event: WorkflowEvent) -> None:
        self.events.append(event)


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=[TextContent(text=text)], stop_reason=StopReason.END_TURN)


def _tool_use(tool_id: str, tool_name: str, input: dict[str, Any] = {}) -> LLMResponse:
    return LLMResponse(
        content=[ToolUseContent(id=tool_id, name=tool_name, input=input)],
        stop_reason=StopReason.TOOL_USE,
    )


def _load(workflow_dir_name: str) -> tuple[WorkflowRegistry, AgentRegistry, ToolRegistry]:
    wf_reg = WorkflowRegistry()
    ag_reg = AgentRegistry()
    tl_reg = ToolRegistry()
    ConfigLoader(wf_reg, ag_reg, tl_reg).load_one(_WORKFLOWS_DIR / workflow_dir_name)
    return wf_reg, ag_reg, tl_reg


def _context(
    wf_reg: WorkflowRegistry,
    ag_reg: AgentRegistry,
    tl_reg: ToolRegistry,
    workflow_id: str,
    run_id: str = "integration-run-001",
) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        workflow_definition=wf_reg.get(workflow_id),
        shared_state=None,
        workflow_registry=wf_reg,
        agent_registry=ag_reg,
        tool_registry=tl_reg,
        memory_store=InMemoryStore(),
        policy_engine=PolicyEngine(),
        observer=_CapturingObserver(),
    )


# ---------------------------------------------------------------------------
# TestIncidentCommanderWorkflow
# ---------------------------------------------------------------------------


class TestIncidentCommanderWorkflow:
    """Exercises the parallel_specialist pattern with three specialists + reviewer."""

    async def test_produces_workflow_result(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        llm = MockLLMProvider([
            _text("High CPU (94%) and memory pressure (87%) detected."),   # metrics_agent
            _text("DB connection timeouts in payment-service."),            # logs_agent
            _text("payment-service v2.4.1 deployed 37 min before spike."), # deployment_agent
            _text("Root cause: DB pool exhaustion post v2.4.1 deploy."),   # reviewer_agent
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Production alert: payment-service p99 latency >2s",
        )
        assert isinstance(result, WorkflowResult)
        assert result.workflow_id == "incident_commander"
        assert result.output == "Root cause: DB pool exhaustion post v2.4.1 deploy."

    async def test_all_specialist_results_included(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        llm = MockLLMProvider([
            _text("Metrics: CPU 94%, latency spike."),
            _text("Logs: circuit breaker opened."),
            _text("Deploy: v2.4.1 shipped 37m ago."),
            _text("Summary: rollback v2.4.1 immediately."),
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Production incident",
        )
        assert len(result.agent_results) == 4
        specialist_ids = {r.agent_id for r in result.agent_results[:3]}
        assert specialist_ids == {"metrics_agent", "logs_agent", "deployment_agent"}
        assert result.agent_results[3].agent_id == "reviewer_agent"

    async def test_concatenated_sections_passed_to_reviewer(self) -> None:
        # The reviewer's output is the final WorkflowResult.output, confirming
        # that the reviewer ran after aggregation.
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        reviewer_output = "Remediation: rollback, scale DB connections, page on-call."
        llm = MockLLMProvider([
            _text("Metrics finding."),
            _text("Logs finding."),
            _text("Deployment finding."),
            _text(reviewer_output),
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Incident",
        )
        assert result.output == reviewer_output

    async def test_metrics_agent_executes_mock_tool(self) -> None:
        # MockLLMProvider returns a tool-use response for the metrics agent,
        # verifying that the MockAdapter registered from tools.yaml is invoked.
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        llm = MockLLMProvider([
            # metrics_agent: tool call then final answer
            _tool_use("call-001", "mock_metrics_tool", {"time_range": "last_5m"}),
            _text("CPU at 94% — critical threshold breached."),
            # remaining agents: direct answers
            _text("Logs: repeated DB timeouts."),
            _text("Deploy: v2.4.1 is the likely culprit."),
            _text("Root cause confirmed: DB pool exhaustion."),
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Production incident",
        )
        assert result.output == "Root cause confirmed: DB pool exhaustion."
        metrics_result = next(
            r for r in result.agent_results if r.agent_id == "metrics_agent"
        )
        assert metrics_result.tool_calls_made == 1


# ---------------------------------------------------------------------------
# TestCustomerSupportWorkflow
# ---------------------------------------------------------------------------


class TestCustomerSupportWorkflow:
    """Exercises the router pattern with a classifier and two specialist branches."""

    async def test_routes_billing_query_to_billing_agent(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([
            _text("billing"),                                   # classifier
            _text("Your invoice was issued on the 1st."),       # billing_agent
        ])
        result = await RouterExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "customer_support"),
            "Why was I charged twice this month?",
        )
        assert result.output == "Your invoice was issued on the 1st."
        assert result.agent_results[0].agent_id == "classifier_agent"
        assert result.agent_results[1].agent_id == "billing_agent"

    async def test_routes_technical_query_to_technical_agent(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([
            _text("technical"),                                         # classifier
            _text("1. Check your API key. 2. Verify endpoint URL."),    # technical_agent
        ])
        result = await RouterExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "customer_support"),
            "I keep getting 401 errors from the API.",
        )
        assert result.output == "1. Check your API key. 2. Verify endpoint URL."
        assert result.agent_results[1].agent_id == "technical_agent"

    async def test_agent_results_contains_classifier_and_target(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([_text("billing"), _text("Refund processed.")])
        result = await RouterExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "customer_support"),
            "I need a refund.",
        )
        assert len(result.agent_results) == 2

    async def test_unknown_route_raises_pattern_execution_error(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([_text("shipping")])  # not in routes
        with pytest.raises(PatternExecutionError, match="shipping"):
            await RouterExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "customer_support"),
                "Where is my package?",
            )


# ---------------------------------------------------------------------------
# TestResearchWorkflow
# ---------------------------------------------------------------------------


class TestResearchWorkflow:
    """Exercises the planner_executor_observer pattern with loop control signals."""

    async def test_done_on_first_iteration(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("research_workflow")
        llm = MockLLMProvider([
            _text("Search for recent papers on transformer attention mechanisms."),  # planner
            _text("Found 5 key papers from 2023 covering sparse attention."),        # executor
            _text("DONE"),                                                           # observer
        ])
        result = await PlannerExecutorObserverExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "research_workflow"),
            "Summarize recent advances in transformer attention mechanisms.",
        )
        assert isinstance(result, WorkflowResult)
        assert result.output == "Found 5 key papers from 2023 covering sparse attention."
        assert len(result.agent_results) == 3

    async def test_continues_on_retry_then_done(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("research_workflow")
        llm = MockLLMProvider([
            _text("Search for transformer papers."),            # planner iter 0
            _text("Results were too broad."),                   # executor iter 0
            _text("RETRY"),                                     # observer iter 0
            _text("Narrow search to sparse attention only."),   # planner iter 1
            _text("Found 3 precise papers on sparse attention."),# executor iter 1
            _text("DONE"),                                      # observer iter 1
        ])
        result = await PlannerExecutorObserverExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "research_workflow"),
            "Summarize sparse attention research.",
        )
        assert result.output == "Found 3 precise papers on sparse attention."
        assert len(result.agent_results) == 6  # 2 iterations × 3 agents

    async def test_max_iterations_exceeded_raises(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("research_workflow")
        # Override max_iterations by modifying pattern_config on the loaded definition
        wf_def = wf_reg.get("research_workflow")
        wf_def.pattern_config["max_iterations"] = 2

        llm = MockLLMProvider([
            _text("Step A."), _text("Result A."), _text("RETRY"),   # iter 0
            _text("Step B."), _text("Result B."), _text("RETRY"),   # iter 1
        ])
        with pytest.raises(PatternExecutionError, match="exceeded"):
            await PlannerExecutorObserverExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "research_workflow"),
                "Research topic.",
            )
