"""Models for the V3 planner: capability descriptors and goal analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums shared across planner phases
# ---------------------------------------------------------------------------


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
# Goal analysis (output of GoalAnalyzer — only LLM step)
# ---------------------------------------------------------------------------


@dataclass
class GoalAnalysis:
    """Structured intent extracted from a natural-language user goal.

    Produced by GoalAnalyzer via a single LLM call. All downstream planner
    steps (agent selection, tool selection, pattern selection) are deterministic
    and consume this struct as input.

    missing_capabilities: capabilities the LLM requested that have no agent
    in the registry (anti-hallucination filter output). Populated by the
    GoalAnalyzer; empty until Phase C.
    """

    required_capabilities: list[str]
    risk_level: RiskLevel
    confidence: float
    reasoning: str
    constraints: list[str]
    requires_hitl: bool
    missing_capabilities: list[str] = field(default_factory=list)


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

    Invariant (Phase B):
        selected_agents == [r.id for r in runtime_agents if not r.generated]

    PlanBuilder is the sole component that constructs and populates both
    selected_agents and runtime_agents. No other component mutates either field.
    selected_agents contains only static (executable) agent IDs; runtime_agents
    contains the full planning result including generated agent definitions.
    """

    plan_id: str
    user_goal: str
    goal_analysis: GoalAnalysis
    selected_pattern: str
    selected_agents: list[str]
    selected_tools: list[str]
    guardrails: list[GuardrailConfig]
    hitl_required: bool
    warnings: list[str]
    explanation: str
    estimated_complexity: str
    estimated_duration_seconds: int
    task_label: str = ""
    runtime_agents: list["RuntimeAgentDefinition"] = field(default_factory=list)


class OperationType(str, Enum):
    READ    = "read"     # safe, no side effects
    WRITE   = "write"    # external side effects
    SEARCH  = "search"   # read-only semantic retrieval
    EXECUTE = "execute"  # runs a subprocess or remote command


@dataclass
class AgentCapabilityDescriptor:
    """Describes what an existing registered agent can do.

    consumes: data tokens this agent needs as input from other agents.
    produces: data tokens this agent outputs for downstream agents to consume.
    Both default to [] for agents without defined contracts.
    """

    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    input_description: str = ""
    output_description: str = ""
    required_tool_capabilities: list[str] = field(default_factory=list)
    consumes: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Runtime agent generation (Phase A — Dynamic Agent Generation)
# ---------------------------------------------------------------------------


@dataclass
class RuntimeAgentDefinition:
    """Unified agent reference produced by RuntimeAgentGenerator.

    generated=False: a static registered agent to reuse; `id` is the registry
    agent_id and `system_prompt` is empty — the runtime loads the full definition
    by ID with no change to the V2 runtime.

    generated=True: a deterministically generated agent for an unregistered
    capability; `system_prompt` and `tool_names` are fully populated and no
    LLM is involved in their construction.

    Closely mirrors AgentDefinition so Phase B can convert with minimal code.
    """

    id: str
    name: str
    description: str
    capabilities: list[str]
    tool_names: list[str]
    system_prompt: str
    generated: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "tool_names": list(self.tool_names),
            "system_prompt": self.system_prompt,
            "generated": self.generated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeAgentDefinition":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            capabilities=list(data["capabilities"]),
            tool_names=list(data["tool_names"]),
            system_prompt=data["system_prompt"],
            generated=bool(data["generated"]),
        )


@dataclass
class PatternCapabilityDescriptor:
    """Describes an execution pattern and when it should be used.

    trigger_capabilities: agent capability tags that indicate this pattern should
    be selected. PatternSelector checks these in priority order; a pattern with
    an empty list matches last as the universal default.
    """

    pattern: str
    name: str
    description: str
    best_for: list[str]
    supported_task_types: list[str]
    requires_reviewer: bool = False
    supports_iteration: bool = False
    min_agents: int = 1
    max_agents: int = 10
    trigger_capabilities: list[str] = field(default_factory=list)
