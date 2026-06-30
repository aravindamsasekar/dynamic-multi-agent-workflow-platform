"""Models for the V3 planner: capability descriptors and goal analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums shared across planner phases
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    CODE_REVIEW = "code_review"
    UNSUPPORTED = "unsupported"  # V3.2+: incident_triage, research, data_analysis


class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Planner-level exception
# ---------------------------------------------------------------------------


class PlannerError(Exception):
    """Raised when the planner cannot produce a valid result.

    Covers: LLM parse failures, schema validation errors, unsupported goals.
    """


# ---------------------------------------------------------------------------
# Goal analysis (output of GoalAnalyzer — only LLM step in V3.1)
# ---------------------------------------------------------------------------


@dataclass
class GoalAnalysis:
    """Structured intent extracted from a natural-language user goal.

    Produced by GoalAnalyzer via a single LLM call. All downstream planner
    steps (agent selection, tool selection, pattern selection) are deterministic
    and consume this struct as input.
    """

    task_type: TaskType
    required_capabilities: list[str]   # capability tags from CapabilityRegistry
    risk_level: RiskLevel
    confidence: float                  # 0.0–1.0
    reasoning: str                     # one-sentence explanation of the classification
    constraints: list[str]             # e.g. ["read_only", "no_external_writes"]
    requires_hitl: bool


# ---------------------------------------------------------------------------
# Guardrail configuration (serialisable; instantiated at execution time)
# ---------------------------------------------------------------------------


@dataclass
class GuardrailConfig:
    """Serialisable description of a policy rule to apply at execution time."""

    rule_type: str              # "content_filter" | "tool_permission"
    config: dict                # rule-specific parameters
    reason: str                 # human-readable rationale


# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    code: str       # e.g. "UNSUPPORTED_TASK_TYPE", "MISSING_AGENT", "CONFIDENCE_TOO_LOW"
    message: str


@dataclass
class ValidationWarning:
    code: str       # e.g. "LOW_CONFIDENCE", "CAPABILITY_UNMATCHED", "HITL_RECOMMENDED"
    message: str


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationWarning]


# ---------------------------------------------------------------------------
# Generated workflow plan (output of PlanBuilder)
# ---------------------------------------------------------------------------


@dataclass
class GeneratedWorkflowPlan:
    """Deterministic plan produced from a GoalAnalysis.

    Contains all selections (pattern, agents, tools) and guardrail configs
    needed to execute the workflow. Does not execute anything — execution
    is triggered separately after user approval (Phase E).
    """

    plan_id: str                           # UUID
    user_goal: str                         # original natural-language goal
    goal_analysis: GoalAnalysis            # structured intent from GoalAnalyzer
    selected_pattern: str                  # "parallel_specialist" | "" for unsupported
    selected_agents: list[str]             # agent_ids from CapabilityRegistry
    selected_tools: list[str]              # tool_names from CapabilityRegistry
    guardrails: list[GuardrailConfig]      # serialisable rule configs
    hitl_required: bool                    # from analysis.requires_hitl
    warnings: list[str]                    # builder-side informational notes
    explanation: str                       # human-readable plan summary
    estimated_complexity: str             # "low" | "medium" | "high" — derived from agent/tool count
    estimated_duration_seconds: int       # rough wall-clock estimate


class OperationType(str, Enum):
    READ    = "read"     # safe, no side effects
    WRITE   = "write"    # external side effects
    SEARCH  = "search"   # read-only semantic retrieval
    EXECUTE = "execute"  # runs a subprocess or remote command


@dataclass
class AgentCapabilityDescriptor:
    """Describes what an existing registered agent can do."""

    agent_id: str
    name: str
    description: str
    capabilities: list[str]             # e.g. ["fetch_pr_data", "fetch_github_diff"]
    supported_task_types: list[str]     # e.g. ["code_review"]
    input_description: str = ""
    output_description: str = ""
    required_tool_capabilities: list[str] = field(default_factory=list)


@dataclass
class ToolCapabilityDescriptor:
    """Describes what a registered tool can do and its risk characteristics."""

    tool_name: str
    name: str
    description: str
    capabilities: list[str]             # e.g. ["read_github_pr", "fetch_pr_metadata"]
    operation_type: OperationType
    data_source: str                    # "github" | "knowledge" | "mcp" | "custom"
    requires_credentials: bool = False
    requires_mcp: bool = False
    is_destructive: bool = False
    requires_hitl: bool = False


@dataclass
class PatternCapabilityDescriptor:
    """Describes an execution pattern and when it should be used."""

    pattern: str                        # matches PatternType enum value string
    name: str
    description: str
    best_for: list[str]                 # use-case tags e.g. ["multi_dimension_analysis"]
    supported_task_types: list[str]     # task types this pattern handles
    requires_reviewer: bool = False
    supports_iteration: bool = False
    min_agents: int = 1
    max_agents: int = 10
