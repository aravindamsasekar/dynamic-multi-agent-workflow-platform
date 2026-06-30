"""Capability Registry — static lookup of what the platform can do."""

from __future__ import annotations

from platform.planner.models import (
    AgentCapabilityDescriptor,
    OperationType,
    PatternCapabilityDescriptor,
    ToolCapabilityDescriptor,
)


class DuplicateCapabilityError(Exception):
    """Raised when a descriptor with the same ID is registered twice."""


class CapabilityRegistry:
    """Holds descriptors for all agents, tools, and patterns the planner can use.

    Loaded once at startup from static data. Used as lookup tables by the
    selectors (Phase C) and as prompt context for the Goal Analyzer (Phase B).
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentCapabilityDescriptor] = {}
        self._tools: dict[str, ToolCapabilityDescriptor] = {}
        self._patterns: dict[str, PatternCapabilityDescriptor] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, descriptor: AgentCapabilityDescriptor) -> None:
        if descriptor.agent_id in self._agents:
            raise DuplicateCapabilityError(
                f"Agent already registered: {descriptor.agent_id!r}"
            )
        self._agents[descriptor.agent_id] = descriptor

    def register_tool(self, descriptor: ToolCapabilityDescriptor) -> None:
        if descriptor.tool_name in self._tools:
            raise DuplicateCapabilityError(
                f"Tool already registered: {descriptor.tool_name!r}"
            )
        self._tools[descriptor.tool_name] = descriptor

    def register_pattern(self, descriptor: PatternCapabilityDescriptor) -> None:
        if descriptor.pattern in self._patterns:
            raise DuplicateCapabilityError(
                f"Pattern already registered: {descriptor.pattern!r}"
            )
        self._patterns[descriptor.pattern] = descriptor

    # ------------------------------------------------------------------
    # Point lookups
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> AgentCapabilityDescriptor | None:
        return self._agents.get(agent_id)

    def get_tool(self, tool_name: str) -> ToolCapabilityDescriptor | None:
        return self._tools.get(tool_name)

    def get_pattern(self, pattern: str) -> PatternCapabilityDescriptor | None:
        return self._patterns.get(pattern)

    # ------------------------------------------------------------------
    # Capability tag queries
    # ------------------------------------------------------------------

    def find_agents_by_capability(self, capability: str) -> list[AgentCapabilityDescriptor]:
        return [d for d in self._agents.values() if capability in d.capabilities]

    def find_tools_by_capability(self, capability: str) -> list[ToolCapabilityDescriptor]:
        return [d for d in self._tools.values() if capability in d.capabilities]

    # ------------------------------------------------------------------
    # Task type queries
    # ------------------------------------------------------------------

    def find_agents_by_task_type(self, task_type: str) -> list[AgentCapabilityDescriptor]:
        return [d for d in self._agents.values() if task_type in d.supported_task_types]

    def find_patterns_for_task_type(self, task_type: str) -> list[PatternCapabilityDescriptor]:
        return [d for d in self._patterns.values() if task_type in d.supported_task_types]

    def get_default_pattern_for_task_type(
        self, task_type: str
    ) -> PatternCapabilityDescriptor | None:
        matches = self.find_patterns_for_task_type(task_type)
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # LLM prompt context
    # ------------------------------------------------------------------

    def to_prompt_summary(self) -> str:
        """Compact text representation of registered capabilities for LLM context."""
        lines: list[str] = []

        lines.append("=== Registered Agents ===")
        for d in self._agents.values():
            caps = ", ".join(d.capabilities)
            lines.append(f"  {d.agent_id} [{', '.join(d.supported_task_types)}]: {caps}")

        lines.append("=== Registered Tools ===")
        for d in self._tools.values():
            caps = ", ".join(d.capabilities)
            lines.append(f"  {d.tool_name} [{d.operation_type.value}]: {caps}")

        lines.append("=== Execution Patterns ===")
        for d in self._patterns.values():
            task_types = ", ".join(d.supported_task_types)
            lines.append(f"  {d.pattern} (task_types: {task_types}): {d.description}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory: pre-loaded with PR review descriptors
    # ------------------------------------------------------------------

    @classmethod
    def build_pr_review_registry(cls) -> "CapabilityRegistry":
        """Returns a registry pre-loaded with all V3.1 PR review capabilities."""
        registry = cls()

        # Agents
        registry.register_agent(AgentCapabilityDescriptor(
            agent_id="pr_data_agent",
            name="PR Data Agent",
            description="Fetches all GitHub data needed to review a pull request.",
            capabilities=["fetch_pr_data", "fetch_github_diff", "fetch_changed_files"],
            supported_task_types=["code_review"],
            input_description="owner, repo, pull_number",
            output_description="PR metadata, changed files list, unified diff",
            required_tool_capabilities=["read_github_pr", "read_github_files", "read_github_diff"],
        ))

        registry.register_agent(AgentCapabilityDescriptor(
            agent_id="review_specialist",
            name="Code Review Specialist",
            description="Reviews code quality, architecture, and maintainability.",
            capabilities=["review_code_quality", "assess_architecture", "check_standards"],
            supported_task_types=["code_review"],
            input_description="PR description and diff",
            output_description="Code quality, architecture, and standards review",
            required_tool_capabilities=["read_github_diff", "search_knowledge"],
        ))

        registry.register_agent(AgentCapabilityDescriptor(
            agent_id="risk_specialist",
            name="Risk Assessment Specialist",
            description="Assesses security, testing, reliability, and performance risks.",
            capabilities=["assess_security", "assess_testing", "assess_reliability"],
            supported_task_types=["code_review"],
            input_description="PR description and diff",
            output_description="Risk assessment across security, testing, reliability, performance",
            required_tool_capabilities=["read_github_diff", "search_knowledge"],
        ))

        registry.register_agent(AgentCapabilityDescriptor(
            agent_id="synthesis_agent",
            name="Synthesis Agent",
            description="Synthesizes specialist findings into a final PR review report.",
            capabilities=["synthesize_findings", "produce_final_report"],
            supported_task_types=["code_review"],
            input_description="Outputs from pr_data_agent, review_specialist, risk_specialist",
            output_description="Structured final PR review with verdict",
            required_tool_capabilities=["search_knowledge", "read_pr_comments"],
        ))

        # Tools
        registry.register_tool(ToolCapabilityDescriptor(
            tool_name="github_get_pr",
            name="GitHub Get PR",
            description="Fetches pull request metadata including title, body, state, and author.",
            capabilities=["read_github_pr", "fetch_pr_metadata"],
            operation_type=OperationType.READ,
            data_source="github",
            requires_credentials=True,
        ))

        registry.register_tool(ToolCapabilityDescriptor(
            tool_name="github_get_files",
            name="GitHub Get Files",
            description="Lists files changed in the pull request with stats.",
            capabilities=["read_github_files", "fetch_changed_files"],
            operation_type=OperationType.READ,
            data_source="github",
            requires_credentials=True,
        ))

        registry.register_tool(ToolCapabilityDescriptor(
            tool_name="github_get_diff",
            name="GitHub Get Diff",
            description="Returns the raw unified diff for all changed files.",
            capabilities=["read_github_diff", "fetch_pr_diff"],
            operation_type=OperationType.READ,
            data_source="github",
            requires_credentials=True,
        ))

        registry.register_tool(ToolCapabilityDescriptor(
            tool_name="knowledge_search",
            name="Knowledge Search",
            description="Searches the knowledge base for coding standards and guidelines.",
            capabilities=["search_knowledge", "search_coding_standards", "search_architecture"],
            operation_type=OperationType.SEARCH,
            data_source="knowledge",
        ))

        registry.register_tool(ToolCapabilityDescriptor(
            tool_name="mcp_get_pr_comments",
            name="MCP Get PR Comments",
            description="Retrieves existing review comments on a pull request via GitHub MCP.",
            capabilities=["read_pr_comments", "fetch_review_threads"],
            operation_type=OperationType.READ,
            data_source="mcp",
            requires_credentials=True,
            requires_mcp=True,
        ))

        # Patterns
        registry.register_pattern(PatternCapabilityDescriptor(
            pattern="parallel_specialist",
            name="Parallel Specialist",
            description=(
                "Runs multiple specialist agents in parallel, then synthesizes their outputs."
            ),
            best_for=["multi_dimension_analysis", "independent_parallel_perspectives", "code_review"],
            supported_task_types=["code_review"],
            requires_reviewer=True,
            supports_iteration=False,
            min_agents=2,
            max_agents=5,
        ))

        registry.register_pattern(PatternCapabilityDescriptor(
            pattern="router",
            name="Router",
            description="Classifies input and routes to the appropriate specialist agent.",
            best_for=["classification", "routing", "support_triage"],
            supported_task_types=["support"],
            requires_reviewer=False,
            supports_iteration=False,
            min_agents=2,
            max_agents=10,
        ))

        registry.register_pattern(PatternCapabilityDescriptor(
            pattern="planner_executor_observer",
            name="Planner-Executor-Observer",
            description=(
                "Iterative pattern for research and exploration with unknown number of steps."
            ),
            best_for=["iterative_research", "exploration", "self_correction"],
            supported_task_types=["research"],
            requires_reviewer=False,
            supports_iteration=True,
            min_agents=3,
            max_agents=3,
        ))

        return registry
