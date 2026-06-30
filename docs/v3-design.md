# V3 — Dynamic Workflow Builder: Architecture Plan

**Status:** V3.1 Complete (919 tests passing)  
**Prerequisite:** V2 complete (684 tests passing, v2.0.0 tagged)  
**Constraint:** Do not modify V2 runtime. All generation is a layer above.  
**V3.1 scope:** One supported goal type ("Review this GitHub PR"). Select from existing registered agents only. No dynamic agent generation.

---

## 0. Design principles

1. **One LLM call per planning request.** The Goal Analyzer makes a single structured LLM call. Every downstream step — agent selection, tool selection, pattern selection, guardrail generation, HITL detection, validation — is deterministic Python. This keeps the system testable and predictable.

2. **Reuse `WorkflowDefinition` as the execution contract.** The planner's output is a `WorkflowDefinition` (existing model). The V2 execution engine receives the same struct it always has. No new execution paths.

3. **`GeneratedWorkflowPlan` wraps `WorkflowDefinition`.** It adds goal, confidence, reasoning, validation results, and approval status as metadata. It does not replace anything.

4. **Fail safe.** Low-confidence plans raise warnings. High-risk plans auto-enable HITL. Unknown tools block plan approval. The planner is never allowed to produce an executable but dangerous plan silently.

5. **V3.1 selects agents from existing registered entries only.** The planner picks agents from the Capability Registry — agents that already exist as YAML definitions on disk. Dynamic agent generation (synthesizing new agents from templates) is a later enhancement and is not part of this prototype.

---

## 1. Capability Registry

### Purpose

A static, structured description of everything the platform can do. The Goal Analyzer receives this as context when analyzing a user goal. It answers: "Given this goal, which capabilities do I need?"

### Not a runtime system

The Capability Registry is not a service and not queried at LLM call time. It is a data structure, loaded once at startup, used as prompt context for the Goal Analyzer and as lookup tables for the selectors.

### Agent capability descriptors

```python
@dataclass
class AgentCapabilityDescriptor:
    agent_id: str                    # "review_specialist" | "pr_data_agent" | ...
    name: str                        # Human-readable
    capabilities: list[str]          # ["review_code_quality", "assess_architecture"]
    input_description: str           # "PR owner/repo/pull_number + diff + knowledge results"
    output_description: str          # "Structured review: Code Quality, Architecture, Standards"
    required_tool_capabilities: list[str]   # ["read_github_diff", "search_knowledge"]
```

Descriptors are populated at startup from existing `agents.yaml` files plus a `capabilities.yaml` tag file in each workflow directory. Every descriptor maps to a real, on-disk agent definition. There are no template or synthetic entries in V3.1.

**V3.1 registered agents for PR review:**

| Agent ID | Capabilities |
|---|---|
| `pr_data_agent` | `fetch_pr_data`, `fetch_github_diff`, `fetch_changed_files` |
| `review_specialist` | `review_code_quality`, `assess_architecture`, `check_standards` |
| `risk_specialist` | `assess_security`, `assess_testing`, `assess_reliability` |
| `synthesis_agent` | `synthesize_findings`, `produce_final_report` |

Dynamic agent generation (creating new `AgentDefinition` objects from templates) is reserved for a later V3 enhancement.

### Tool capability descriptors

```python
@dataclass
class ToolCapabilityDescriptor:
    tool_name: str                   # "github_get_diff" | "knowledge_search" | ...
    capabilities: list[str]          # ["read_github_diff", "read_file_changes"]
    operation_type: OperationType    # READ | WRITE | SEARCH | EXECUTE
    data_source: str                 # "github" | "knowledge" | "filesystem" | "custom"
    requires_credentials: bool       # False for public GitHub, True for token-gated ops
    requires_mcp: bool               # True if tool goes through MCPAdapter
    is_destructive: bool             # True for delete/post/send operations
    requires_hitl: bool              # True if this tool should always trigger HITL

class OperationType(str, Enum):
    READ    = "read"      # safe, no side effects
    WRITE   = "write"     # external side effects — requires explicit approval
    SEARCH  = "search"    # read-only semantic retrieval
    EXECUTE = "execute"   # runs a subprocess or remote command
```

**MCP tool representation:** MCP tools are discovered dynamically via `MCPConnectionManager.list_tools()`. The Capability Registry holds a descriptor for each *configured* MCP tool (those already in `tools.yaml` files), and a mechanism to declare new MCP servers by providing `server_command` + `server_args`. At plan time, if the planner needs an MCP capability not currently in the registry, it can emit a "tool suggestion" that the user must approve manually.

```python
@dataclass
class MCPServerDescriptor:
    server_id: str                   # "github-mcp" | "filesystem-mcp"
    server_command: str              # "npx"
    server_args: list[str]           # ["-y", "@modelcontextprotocol/server-github"]
    known_tools: list[str]           # pre-known tool names (before connect)
    capabilities: list[str]          # ["read_pr_comments", "post_pr_review"]
```

### Pattern capability descriptors

```python
@dataclass
class PatternCapabilityDescriptor:
    pattern: PatternType
    best_for: list[str]              # ["parallel_analysis", "multi_dimension_review"]
    requires_reviewer: bool
    supports_iteration: bool
    complexity: str                  # "simple" | "moderate" | "complex"
    description: str
```

| Pattern | Best for | Reviewer needed |
|---|---|---|
| `parallel_specialist` | Multi-dimension analysis, independent parallel perspectives | Yes (synthesis) |
| `router` | Classification + specialized handling, mutually exclusive branches | No |
| `planner_executor_observer` | Iterative research, unknown number of steps, self-correcting loops | No (observer is controller) |

---

## 2. Goal Analyzer

### Role

Converts a natural language user goal into a structured `GoalAnalysis`. This is the **only LLM call** in the planning phase.

### V3.1 scope

In V3.1, only `CODE_REVIEW` goals are supported. If the LLM determines the goal does not match a PR review, it returns `task_type = UNSUPPORTED` with `confidence = 0.0`, and the API returns a 400 with a clear message: "This goal type is not yet supported. Supported goal: Review a GitHub PR." The `UNSUPPORTED` path requires no further planning — it short-circuits immediately after the Goal Analyzer returns. Additional goal types (incident triage, research, etc.) are added in V3.2+.

### Input / output schema

```python
@dataclass
class GoalContext:
    goal: str                        # "Review this GitHub PR for architecture, testing and security"
    context: dict[str, Any]          # {"owner": "...", "repo": "...", "pull_number": 42}
    available_capabilities: list[str]  # capability names from CapabilityRegistry (prompt context)

@dataclass
class GoalAnalysis:
    task_type: TaskType              # CODE_REVIEW | UNSUPPORTED (V3.1 only)
    summary: str                     # "Review PR #42 for architecture quality, test coverage, and security"
    required_capabilities: list[str] # ["fetch_pr_data", "review_code_quality", "assess_security", "synthesize_findings"]
    data_sources: list[str]          # ["github", "knowledge"]
    constraints: list[str]           # ["read_only", "no_external_writes"]
    risk_level: RiskLevel            # LOW | MEDIUM | HIGH | CRITICAL
    confidence: float                # 0.0–1.0
    reasoning: str                   # "Goal clearly maps to PR review pattern."
    unknown_capabilities: list[str]  # capabilities requested but not available in registry

class TaskType(str, Enum):
    CODE_REVIEW  = "code_review"
    UNSUPPORTED  = "unsupported"   # V3.1: all non-PR-review goals land here
    # Future: INCIDENT_TRIAGE, RESEARCH, SUPPORT, DATA_ANALYSIS (V3.2+)

class RiskLevel(str, Enum):
    LOW      = "Low"
    MEDIUM   = "Medium"
    HIGH     = "High"
    CRITICAL = "Critical"
```

### LLM call design

The Goal Analyzer sends ONE message to the LLM. It uses structured output (JSON mode or tool_use) so the response is always parseable.

**System prompt:**
```
You are a workflow planning assistant. Your job is to analyze a user's goal and extract
structured intent that can be used to build an automated multi-agent workflow.

Available capabilities in this platform:
{capability_registry_summary}   ← injected at call time

Return a JSON object matching this exact schema: {schema}

Be conservative:
- Set confidence < 0.7 if the goal is ambiguous or requires capabilities not listed above.
- Set risk_level = "High" if the goal involves writing to external systems, production changes,
  or financial operations. Set "Critical" if it involves deletion or irreversible actions.
- List unknown_capabilities if the user asks for something the platform cannot do.
```

**capability_registry_summary** is a compact text representation (~500 tokens) of all registered agents, tools, and patterns — sufficient for the LLM to make matching decisions.

### Structured output

The analyzer validates the JSON response against the `GoalAnalysis` schema. If parsing fails, it retries once with an explicit error message. If it fails twice, it returns a `GoalAnalysis` with `confidence=0.0` and `task_type=CUSTOM`, which will surface as warnings in the plan.

### Testing

The `GoalAnalyzer` accepts an `ILLMProvider` injection, so tests use `MockLLMProvider` with pre-queued structured responses. The analyzer is never tested against a real LLM in the test suite.

---

## 3. Agent Selection

### Logic (V3.1 — registry selection only)

```
AgentSelector.select(analysis: GoalAnalysis, capability_registry)
  → list[str]     ← agent_ids of existing registered agents
```

The selector matches each required capability from `GoalAnalysis.required_capabilities` to the best-matching `AgentCapabilityDescriptor` in the Capability Registry. It returns agent IDs, not `AgentDefinition` objects — the definitions already exist on disk and are loaded by `AgentRegistry` at startup as usual.

**Step 1 — For each required capability, find the registered agent that covers it.**

```python
def select(analysis: GoalAnalysis, registry: CapabilityRegistry) -> list[str]:
    selected: list[str] = []
    unmatched: list[str] = []
    for capability in analysis.required_capabilities:
        match = registry.find_agent_by_capability(capability)
        if match and match.agent_id not in selected:
            selected.append(match.agent_id)
        elif not match:
            unmatched.append(capability)
    return selected, unmatched
```

**Step 2 — Surface unmatched capabilities.**
If any required capability has no registered agent, it is added to `ValidationResult.warnings` with code `CAPABILITY_UNMATCHED`. This is non-blocking unless every capability is unmatched (which produces an error).

**Step 3 — No agent generation in V3.1.**
If an agent does not exist in the registry, the plan notes the gap and proceeds with what is available. Dynamic agent generation from templates is a V3.2+ enhancement.

### Agent count guardrail

The selector caps parallel agents at **5**. More than 5 agents is a warning; more than 7 is a blocking error.

---

## 4. Tool Selection

### Three pools

**Pool 1 — Registered tools** (from `ToolRegistry`): Tools already registered at startup. Can be assigned to any agent that needs the matching capability.

**Pool 2 — Knowledge tools**: The `knowledge_search` tool is selected when the analysis identifies knowledge/standards retrieval as a required capability. The selector picks appropriate collections based on `GoalAnalysis.data_sources` and `task_type`:

| Task type | Suggested collections |
|---|---|
| `code_review` | `coding-standards`, `architecture` |
| `incident_triage` | `runbooks`, `architecture` |
| `research` | All collections |

**Pool 3 — MCP tools**: Selected when the analysis requires capabilities only available via MCP (e.g., post PR comments, read filesystem). The selector checks `MCPServerDescriptor.capabilities` in the registry.

### Tool permission model

```python
class ToolPermission(str, Enum):
    ALLOW          = "allow"       # read-only, safe
    REQUIRE_HITL   = "require_hitl"  # write/side-effect — needs approval gate
    BLOCK          = "block"       # destructive — cannot be used without explicit override
```

| Operation type | Default permission |
|---|---|
| READ | ALLOW |
| SEARCH | ALLOW |
| WRITE (non-destructive, e.g. post PR comment) | REQUIRE_HITL |
| WRITE (destructive, e.g. delete, send email) | BLOCK |
| EXECUTE | REQUIRE_HITL |

### Handling unavailable tools

If a required capability maps to no available tool:

1. Mark the capability as `unavailable` in the `GeneratedWorkflowPlan`.
2. Add a warning: "Capability X is not available. Agent Y will operate without it."
3. If the unavailable tool is critical (e.g., the only data source), add an error that blocks plan approval.
4. Surface a `suggestions` field: "To enable this capability, add a tool of type X to your tools.yaml."

The validator catches all tool availability issues before the plan can be approved.

---

## 5. Pattern Selection

The `PatternSelector` is a deterministic function. No LLM involved.

### V3.1 — fixed to `parallel_specialist`

In V3.1, only `CODE_REVIEW` goals are supported, and the PR review workflow uses `parallel_specialist`. Pattern selection is therefore trivially determined:

```python
def select_pattern(analysis: GoalAnalysis) -> PatternType:
    if analysis.task_type == TaskType.CODE_REVIEW:
        return PatternType.parallel_specialist
    raise UnsupportedGoalType(analysis.task_type)
```

### Pattern-specific constraints for `parallel_specialist`

The selector verifies that a synthesis agent (`synthesize_findings` capability) is included in the selected agent list. If it is not found in the registry, a `CAPABILITY_UNMATCHED` warning is added — the validator will block approval if the reviewer is absent.

### Future patterns (V3.2+)

The full decision tree (router for classification, PEO for iterative research) is preserved here for future reference, but is not wired in V3.1:

```
# V3.2+ decision tree (not active in V3.1):
# if SUPPORT or "route" → ROUTER
# if RESEARCH or "investigate" → PLANNER_EXECUTOR_OBSERVER
# if multiple independent dimensions → PARALLEL_SPECIALIST
# else → PARALLEL_SPECIALIST
```

---

## 6. Guardrail Generation

Guardrails are `IRule` instances added to the `PolicyEngine` for a generated workflow. They are generated deterministically from `GoalAnalysis`.

### Rule types

**ContentFilterRule** (already exists): Blocks agent outputs that contain prohibited terms.

```python
# New rule types for V3:

class ToolPermissionRule(IRule):
    """Blocks tool calls to operations above the allowed permission level."""
    def __init__(self, blocked_operations: list[OperationType]) -> None: ...
    def check(self, context: dict[str, Any]) -> PolicyDecision: ...

class ConfidenceThresholdRule(IRule):
    """Blocks execution if planner confidence is below threshold."""
    def __init__(self, min_confidence: float) -> None: ...
    def check(self, context: dict[str, Any]) -> PolicyDecision: ...
```

### Generation logic

```python
def generate_guardrails(analysis: GoalAnalysis, selected_tools: list[ToolCapabilityDescriptor]) -> list[IRule]:
    rules: list[IRule] = []

    # 1. Block destructive terms if any write tools selected
    if analysis.risk_level in (HIGH, CRITICAL):
        rules.append(ContentFilterRule(blocked_terms=[
            "delete", "drop table", "rm -rf", "truncate", "irreversible"
        ]))

    # 2. Block write operations if goal is read-only
    if "read_only" in analysis.constraints or "no_external_writes" in analysis.constraints:
        rules.append(ToolPermissionRule(blocked_operations=[WRITE, EXECUTE]))

    # 3. Block financial terms if not explicitly a finance workflow
    if "money" not in analysis.data_sources and "finance" not in analysis.task_type:
        rules.append(ContentFilterRule(blocked_terms=[
            "transfer funds", "charge card", "issue refund"
        ]))

    # 4. Confidence threshold
    if analysis.confidence < 0.8:
        rules.append(ConfidenceThresholdRule(min_confidence=analysis.confidence))

    return rules
```

### What guardrails cannot do

- They cannot prevent the LLM from *generating* harmful content (only from continuing after generation).
- They do not replace proper access control for production environments.
- They are best-effort safety nets, not a security boundary.

This limitation is documented in the plan preview shown to the user.

---

## 7. HITL Generation

### When to auto-enable HITL

HITL is set to `hitl_enabled: true` in the generated `WorkflowDefinition` if ANY of:

| Condition | Rationale |
|---|---|
| `risk_level in (HIGH, CRITICAL)` | Human review before consequential actions |
| `confidence < 0.70` | Low-confidence plans need human verification |
| Any write tool selected | External side effects need approval |
| Any MCP tool with WRITE or EXECUTE type | MCP operations can have unpredictable reach |
| Goal contains trigger keywords | "production", "deploy", "delete", "send", "post", "notify", "refund" |

### Checkpoint placement

In `parallel_specialist`, HITL pauses execution **after the parallel phase and before the reviewer/synthesis agent**. The human sees all specialist outputs and approves before the final synthesis runs. This is the existing V2 HITL mechanism — no changes needed.

For `planner_executor_observer`, HITL could interrupt after the observer's decision. This is a V3.1 enhancement; V3.0 does not add new HITL hooks to the PEO pattern.

### What HITL surfaces to the user

The approval gate shows:
- All specialist agent outputs (the inputs to the reviewer)
- The generated plan's confidence and risk level
- Which write tools are pending in the reviewer phase
- A structured diff of what will change if approved vs. rejected

This is displayed via the existing `GET /runs/{run_id}` endpoint plus a new `GET /runs/{run_id}/hitl-context` endpoint added in V3.

---

## 8. Workflow Plan Representation

### Models

```python
class PlanStatus(str, Enum):
    PENDING_REVIEW  = "pending_review"
    APPROVED        = "approved"
    REJECTED        = "rejected"
    EXECUTED        = "executed"
    FAILED          = "failed"

class GeneratedWorkflowPlan(BaseModel):
    plan_id: str                              # UUID, stable reference
    goal: str                                 # original user goal text
    context: dict[str, Any]                   # extra context (owner, repo, pull_number, etc.)
    analysis: GoalAnalysis                    # structured intent from GoalAnalyzer
    workflow_definition: WorkflowDefinition   # REUSES EXISTING MODEL — V2 engine runs this
    guardrails: list[dict[str, Any]]          # serialised rule configs
    hitl_reason: str | None                   # human-readable HITL rationale
    validation: ValidationResult              # filled after WorkflowPlanValidator runs
    status: PlanStatus
    execution_run_id: str | None              # set when approved and run
    created_at: datetime
    updated_at: datetime
```

No `generated_agents` field. All agent IDs in `workflow_definition.agent_ids` refer to agents already registered in `AgentRegistry` at startup. There is nothing ephemeral to manage.

### Key insight: reuse `WorkflowDefinition`

The `workflow_definition` field is a fully-populated `WorkflowDefinition` object. The V2 `Orchestrator.run()` method accepts a `workflow_id` — for dynamic plans, we add `Orchestrator.run_plan(plan: GeneratedWorkflowPlan)` which:
1. Looks up all agent IDs in the existing `AgentRegistry` (no scoped registry needed)
2. Calls the existing pattern executor with the `WorkflowDefinition`

The execution engine sees no difference. It operates on `WorkflowDefinition` + the standard registries exactly as it always has.

### Why not save generated workflows as YAML?

YAML files work for reusable, human-curated workflows. Generated plans are ephemeral by default — they live in SQLite, not the filesystem. If a user wants to "promote" a generated plan to a permanent workflow, they can export it to YAML (a V3.1 feature). This keeps the V2 config system clean.

---

## 9. Workflow Validation

### Validation is fully deterministic

The `WorkflowPlanValidator` has no LLM dependency. It operates on the `GeneratedWorkflowPlan` and the live registries.

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[ValidationError]     # BLOCKING — plan cannot be approved
    warnings: list[ValidationWarning] # NON-BLOCKING — shown to user, plan can proceed
    suggestions: list[str]            # optional improvements

@dataclass  
class ValidationError:
    code: str       # "TOOL_NOT_FOUND" | "AGENT_MISSING_TOOL" | "PATTERN_INCOMPATIBLE" | ...
    message: str
    field: str      # which part of the plan has the error

@dataclass
class ValidationWarning:
    code: str       # "LOW_CONFIDENCE" | "HIGH_AGENT_COUNT" | "WRITE_TOOL_NO_HITL" | ...
    message: str
```

### Validation rules

| Check | Error / Warning | Condition |
|---|---|---|
| Tool existence | ERROR | Any `tool_name` in any agent not found in `ToolRegistry` |
| Agent completeness | ERROR | An `agent_id` in `workflow_definition.agent_ids` has no definition |
| Pattern config | ERROR | `reviewer_agent_id` missing for `parallel_specialist` with multiple agents |
| Capability coverage | WARNING | A required capability from `GoalAnalysis` has no corresponding agent |
| Low confidence | WARNING | `analysis.confidence < 0.70` |
| High agent count | WARNING | More than 5 parallel agents |
| Write tool without HITL | WARNING | A WRITE tool is selected but `hitl_enabled = false` |
| Unknown capabilities | WARNING | `GoalAnalysis.unknown_capabilities` is non-empty |
| Critical risk without HITL | ERROR | `risk_level = CRITICAL` and `hitl_enabled = false` |
| Empty agent list | ERROR | `workflow_definition.agent_ids` is empty |
| Confidence below minimum | ERROR | `analysis.confidence < 0.30` (plan is too uncertain to run) |

### Validation is run automatically

`WorkflowPlanBuilder.build()` always runs the validator before returning the plan. A plan with errors is returned to the API with `status = PENDING_REVIEW` and `is_valid = false`. The approval endpoint (`POST /workflows/generated/{plan_id}/approve`) re-validates before executing and rejects if errors remain.

---

## 10. Preview and Approval API

### New endpoints

```
POST   /workflows/generate
       body: { goal: str, context: dict }
       → GeneratedWorkflowPlanResponse   (plan_id, preview, validation, status)
       Note: generates and stores plan, does NOT execute

GET    /workflows/generated/{plan_id}
       → GeneratedWorkflowPlanResponse   (full plan detail)

POST   /workflows/generated/{plan_id}/approve
       body: {}
       → RunStatusResponse               (same schema as POST /runs/ — immediately starts execution)

POST   /workflows/generated/{plan_id}/edit
       body: PlanEditRequest             (partial edits, e.g. change an agent's tool list)
       → GeneratedWorkflowPlanResponse   (re-validates and returns updated plan)

POST   /workflows/generated/{plan_id}/reject
       body: { reason: str }
       → { plan_id, status: "rejected", reason }

GET    /workflows/generated/
       → list[GeneratedWorkflowPlanSummary]   (list all plans, paginated)
```

### `GeneratedWorkflowPlanResponse` (what the user sees)

```json
{
  "plan_id": "p-3f7a...",
  "status": "pending_review",
  "goal": "Review PR #42 for architecture, testing and security",
  "analysis": {
    "task_type": "code_review",
    "summary": "Multi-dimension PR review: architecture quality, test coverage, and security assessment",
    "risk_level": "Low",
    "confidence": 0.92
  },
  "workflow": {
    "pattern": "parallel_specialist",
    "agents": [
      { "agent_id": "pr_data_agent",    "role": "Data collection",   "tools": ["github_get_pr", "github_get_files", "github_get_diff"] },
      { "agent_id": "review_specialist","role": "Code review",        "tools": ["github_get_diff", "knowledge_search"] },
      { "agent_id": "risk_specialist",  "role": "Risk assessment",   "tools": ["github_get_diff", "knowledge_search"] },
      { "agent_id": "synthesis_agent",  "role": "Final synthesis",   "tools": ["knowledge_search", "mcp_get_pr_comments"] }
    ],
    "hitl_enabled": false,
    "hitl_reason": null
  },
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": [],
    "suggestions": ["Consider enabling HITL if this is a production repository."]
  },
  "guardrails": ["ContentFilter: blocks destructive terms"],
  "reasoning": "Goal maps clearly to the existing pr_review pattern. All required capabilities are covered by registered agents."
}
```

### Edit handling

`PlanEditRequest` supports targeted edits:
- Add/remove tools from an agent
- Replace a generated system prompt
- Enable/disable HITL
- Change the pattern
- Add a guardrail term

After an edit, the plan is re-validated and a new `ValidationResult` is returned. The edit is stored as a diff against the original plan (version history).

---

## 11. Persistence

### New SQLite table

```sql
CREATE TABLE generated_workflow_plans (
    plan_id         TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    context_json    TEXT,                  -- original context dict
    analysis_json   TEXT NOT NULL,         -- GoalAnalysis serialized
    plan_json       TEXT NOT NULL,         -- full GeneratedWorkflowPlan serialized
    status          TEXT NOT NULL,         -- pending_review | approved | rejected | executed
    execution_run_id TEXT,                 -- set when status = executed
    created_at      TIMESTAMP NOT NULL,
    updated_at      TIMESTAMP NOT NULL
);

CREATE TABLE generated_plan_edits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id         TEXT NOT NULL REFERENCES generated_workflow_plans(plan_id),
    edit_json       TEXT NOT NULL,         -- PlanEditRequest serialized
    created_at      TIMESTAMP NOT NULL
);
```

### Versioning

Plans are immutable after approval. Edits before approval create new rows in `generated_plan_edits` and update the `plan_json` + `updated_at`. After approval and execution, the plan is frozen — a new `POST /workflows/generate` must be issued for a new run.

### Reuse as templates

A successful plan (status = executed) can be promoted to a named YAML workflow via:

```
POST /workflows/generated/{plan_id}/export
body: { workflow_id: "my_custom_review", save_to_disk: false }
→ { workflow_yaml, agents_yaml, tools_yaml }
```

`save_to_disk: true` writes the files to `workflows/my_custom_review/` and registers them for future use without going through the planner. This is a V3.1 feature.

---

## 12. Testing Strategy

### Core principle

The only non-deterministic component is `GoalAnalyzer`. Everything else — selectors, guardrail generator, HITL detector, validator — is pure deterministic Python. Test coverage is straightforward.

### Test layers

**Layer 1 — Unit tests (no LLM)**

Each component tested in isolation:

| Component | Test approach | Target count |
|---|---|---|
| `AgentSelector` | Fixed `GoalAnalysis` → assert selected agent IDs from registry | 10 tests |
| `ToolSelector` | Fixed capabilities → assert tool assignments | 10 tests |
| `PatternSelector` | CODE_REVIEW → parallel_specialist; UNSUPPORTED → raises | 5 tests |
| `GuardrailGenerator` | Fixed risk levels / constraints → assert rule types | 10 tests |
| `HITLDetector` | Fixed risk / tools / keywords → assert bool | 10 tests |
| `WorkflowPlanValidator` | Valid and invalid plans → assert error codes | 15 tests |
| `GoalAnalyzer` | `MockLLMProvider` with queued JSON → assert `GoalAnalysis` | 15 tests |

**Layer 2 — Golden plan test**

V3.1 has one supported goal type, so one golden fixture:

```python
GOLDEN_PLANS = [
    ("Review PR #42 for architecture and security",
     "fixtures/golden_pr_review_plan.json"),
    # V3.2+: incident_triage, research, etc.
]
```

The fixture captures the exact `GeneratedWorkflowPlan` structure for the PR review goal. If the plan shape changes intentionally, update and commit the fixture. This catches unintended regression without touching the LLM.

**Layer 3 — Validation tests**

Deliberately broken plans → assert specific `ValidationError` codes.

```python
def test_missing_tool_raises_error():
    plan = build_plan_with_nonexistent_tool("phantom_tool")
    result = WorkflowPlanValidator().validate(plan, tool_registry, agent_registry)
    assert not result.is_valid
    assert any(e.code == "TOOL_NOT_FOUND" for e in result.errors)
```

**Layer 4 — API integration tests**

Full HTTP layer tests using `httpx.AsyncClient` + `MockLLMProvider`:

```python
async def test_generate_then_approve_executes_workflow():
    # POST /workflows/generate
    response = await client.post("/workflows/generate", json={"goal": "...", "context": {}})
    plan_id = response.json()["plan_id"]

    # POST /workflows/generated/{plan_id}/approve
    run_response = await client.post(f"/workflows/generated/{plan_id}/approve")
    assert run_response.json()["status"] == "completed"
```

**Layer 5 — End-to-end tests**

Mock both the Goal Analyzer LLM (planner) and the execution LLM (runtime). These are the most valuable tests: they verify the full flow from goal to executed run_id.

```python
async def test_pr_review_goal_generates_and_executes():
    # Planner LLM returns a specific GoalAnalysis JSON
    planner_llm = MockLLMProvider([_text(PR_REVIEW_GOAL_ANALYSIS_JSON)])
    # Execution LLM drives the agents
    execution_llm = MockLLMProvider([...agent responses...])
    # Both are injected via dependency overrides
    ...
```

### What NOT to test

- Real LLM output quality (too flaky, belongs in evals)
- YAML round-trips for generated plans (implementation detail)
- MCP server subprocess in unit tests (always mock MCPConnectionManager)

---

## 13. Implementation phases

Each phase leaves the repository in a passing state. All existing 684 tests continue to pass throughout. New tests are added with each phase. Stop and wait for approval before starting the next phase.

---

### Phase A — Capability Registry
**Effort:** 2 days  
**Deliverables:**
- `platform/planner/models.py` — `GoalAnalysis`, `GeneratedWorkflowPlan`, `ValidationResult`, `PlanStatus`, `RiskLevel`, `TaskType`
- `platform/planner/capability_registry.py` — `AgentCapabilityDescriptor`, `ToolCapabilityDescriptor`, `MCPServerDescriptor`, `PatternCapabilityDescriptor`, `CapabilityRegistry`
- `workflows/pr_review/capabilities.yaml` — capability tags for pr_review agents
- Static capability data loaded at startup from existing `agents.yaml` + `capabilities.yaml`
- ~15 tests: descriptor construction, capability lookup by capability name, registry serialization to prompt summary

No API changes. No LLM. No selectors yet. Pure data model and loader.

---

### Phase B — Goal Analyzer
**Effort:** 2 days  
**Deliverables:**
- `platform/planner/goal_analyzer.py` — `GoalAnalyzer` class
- Accepts `ILLMProvider` injection (test-friendly)
- Builds compact capability summary from `CapabilityRegistry` for prompt context
- Parses structured JSON response into `GoalAnalysis`
- Returns `task_type = UNSUPPORTED` immediately for non-CODE_REVIEW goals
- Retry once on JSON parse failure; fall back to `confidence = 0.0` on second failure
- ~15 tests: PR review goal → CODE_REVIEW, non-PR goal → UNSUPPORTED, low confidence path, parse failure retry

No API changes. No execution changes.

---

### Phase C — Plan Builder + Validator
**Effort:** 3 days  
**Deliverables:**
- `platform/planner/agent_selector.py` — selects agent IDs from registry (no generation)
- `platform/planner/tool_selector.py` — assigns registered tools by capability
- `platform/planner/pattern_selector.py` — CODE_REVIEW → parallel_specialist
- `platform/planner/guardrail_generator.py` — generates `ContentFilterRule` / `ToolPermissionRule` from risk level
- `platform/planner/hitl_detector.py` — sets `hitl_enabled` flag
- `platform/planner/validator.py` — deterministic validation, all error/warning codes
- `platform/planner/plan_builder.py` — orchestrates all of the above into `GeneratedWorkflowPlan`
- New policy rules: `ToolPermissionRule`, `ConfidenceThresholdRule` in `platform/policy/rules/`
- Golden fixture: `tests/fixtures/golden_pr_review_plan.json`
- ~40 tests: each selector, guardrail/HITL unit tests, validator error codes, golden plan assertion

No API changes. No LLM in selectors/validator. Plan builder output is a `GeneratedWorkflowPlan` ready to be stored and approved.

---

### Phase D — Preview API (generate + get + reject)
**Effort:** 2 days  
**Deliverables:**
- `api/routers/planner.py` — `POST /workflows/generate`, `GET /workflows/generated/{plan_id}`, `GET /workflows/generated/`, `POST /workflows/generated/{plan_id}/reject`
- `api/schemas/planner.py` — Pydantic request/response models
- `platform/persistence/repositories/plan_repo.py` — SQLite CRUD for plans
- SQLite migration: `generated_workflow_plans` table
- ~15 integration tests: generate (PR goal), generate (unsupported goal → 400), retrieve plan, list plans, reject plan

No execution yet. Does not call `Orchestrator.run()`.

---

### Phase E — Approval and Execution
**Effort:** 2 days  
**Deliverables:**
- `POST /workflows/generated/{plan_id}/approve` — re-validates then calls `Orchestrator.run_plan()`
- `Orchestrator.run_plan(plan: GeneratedWorkflowPlan)` — looks up existing agent IDs, calls existing pattern executor unchanged
- ~10 integration tests + ~5 end-to-end tests (mock LLM for both planner and executor)

This is the phase where a generated plan becomes a running workflow execution. The V2 engine is called unchanged.

---

### Summary

| Phase | Area | Effort | New tests |
|---|---|---|---|
| A | Capability Registry | 2 days | ~15 |
| B | Goal Analyzer | 2 days | ~15 |
| C | Plan Builder + Validator | 3 days | ~40 |
| D | Preview API | 2 days | ~15 |
| E | Approval + Execution | 2 days | ~15 |
| **Total** | | **~11 days** | **~100 tests** |

Target test count after V3.1: **~784 tests** (684 existing + ~100 new).

**V3.2+ backlog** (not in this prototype):
- Additional goal types: incident triage, research, data analysis
- Dynamic agent generation from templates
- Plan edit endpoint (`POST /workflows/generated/{plan_id}/edit`)
- Plan export to YAML (`POST /workflows/generated/{plan_id}/export`)
- PEO and router pattern selection
- Batch goals (multiple PRs in one request)

---

## Open questions (resolve before Phase A)

1. **Capability tagging format:** Capability tags for existing agents can live in a separate `capabilities.yaml` per workflow directory, or be added as an optional field directly to each entry in `agents.yaml`. A separate file avoids any schema change to `agents.yaml` and keeps the capability registry independent of the agent config loader. This is the recommended approach for Phase A.

2. **Planner LLM selection:** Should the Goal Analyzer use the same `OPENAI_MODEL` env variable as the agents, or a separate `PLANNER_MODEL`? Separating them allows using a cheaper/faster model for planning. V3.1 uses the same `OPENAI_MODEL` by default; `PLANNER_MODEL` can be added as an override in V3.2.
