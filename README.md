# Dynamic Multi-Agent Workflow Platform

A reusable, configuration-driven multi-agent workflow platform built with Python and FastAPI.

## V1 Scope

### Supported Patterns
- **Parallel Specialist** — Fan out to specialist agents concurrently, aggregate results, optional reviewer pass
- **Router** — Classify input and dispatch to the matched specialist agent
- **Planner → Executor → Observer** — Plan a set of steps, execute each, observe results, loop until done

### Demo Workflows
- **Incident Commander** — Planner → Executor → Observer
- **Customer Support** — Router with HITL approval
- **PR Review** — Parallel Specialist
- **DevOps Remediation** — Planner → Executor → Observer

### Architecture Constraints
- Python 3.11+
- FastAPI
- Clean Architecture — `core` ← `platform` ← `api`
- No LangGraph, CrewAI, or AutoGen
- Custom lightweight execution engine
- Configuration-driven workflows via YAML

## Project Structure

```
platform/   # Reusable platform components — no workflow-specific code
workflows/  # Workflow definitions — YAML config only, no Python
api/        # FastAPI delivery layer — HTTP routing and serialization only
tests/      # Unit and integration tests
```

## Design Principles

- New workflows are added by dropping a YAML folder under `workflows/` — zero platform code changes.
- Platform components are separated from workflow-specific definitions.
- Every component has a clear interface defined in `platform/core/interfaces/`.
- Memory, policies, observability, and tools are all pluggable.

## Getting Started

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Running Tests

```bash
pytest tests/
```
