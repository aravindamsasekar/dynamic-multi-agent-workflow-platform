# Testing Guidelines

## Philosophy

Tests are first-class code. They document intent, catch regressions, and give maintainers confidence to refactor. A codebase without tests is a liability.

## Test Structure

### File Layout

```
tests/
  unit/
    platform/     # Fast unit tests per platform module
    api/          # FastAPI endpoint tests (no network)
    persistence/  # Repository tests (in-memory SQLite)
  integration/    # End-to-end pattern tests with real YAML + MockLLMProvider
```

### Naming Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<ConceptUnderTest>` (e.g., `TestKnowledgeRetriever`)
- Test methods: `test_<scenario>` — use descriptive names that read as sentences
  - Good: `test_empty_query_returns_empty`
  - Bad: `test_1`, `test_case_a`

### Class-based Tests

All tests live in classes. Do not write module-level test functions.

```python
class TestMyComponent:
    async def test_returns_correct_value(self):
        ...
```

## Unit Tests

### Scope

A unit test covers a single class or function in isolation. External dependencies (database, network, file system, LLMs) are mocked or replaced with in-memory equivalents.

### Fixtures

- Use `pytest.fixture` for reusable setup. Prefer function-scoped fixtures (the default).
- Fixtures for database connections must use `StaticPool` with in-memory SQLite:
  ```python
  engine = create_engine(
      "sqlite:///:memory:",
      connect_args={"check_same_thread": False},
      poolclass=StaticPool,
  )
  ```

### Mocking

- Use `unittest.mock.MagicMock` for synchronous interfaces and `AsyncMock` for async ones.
- Prefer `monkeypatch.setenv` over patching `os.environ` directly.
- Use `spec=ClassName` when constructing mocks to catch wrong attribute access at test time.
- Never patch a symbol where it is *defined* — patch it where it is *used*:
  ```python
  # Wrong:
  patch("platform.tools.github_adapter.httpx")
  # Right (patch where used):
  patch("platform.tools.github_adapter.httpx.AsyncClient")
  ```

### Coverage Targets

Every new module added to `platform/` needs unit tests covering:
- The happy path
- Edge cases (empty input, zero values, missing optional fields)
- Error paths (invalid arguments, dependency failures)

## Integration Tests

Integration tests in `tests/integration/` load real workflow YAML files and run the full pattern executor stack with `MockLLMProvider`. They must:
- Use `MockLLMProvider` — no real LLM calls
- Mock external HTTP calls (GitHub, etc.) via `unittest.mock.patch`
- Not depend on environment variables

### MockLLMProvider

Queue responses in the order the pattern executor will consume them. Each response is either a `_text(str)` (final answer) or `_tool_use(id, name, input)` (tool call):

```python
llm = MockLLMProvider([
    _tool_use("t1", "github_get_pr", {...}),
    _text("PR metadata retrieved."),
    _text("Approved."),
])
```

## Async Tests

All async test methods work without any decorator. The `pyproject.toml` has:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

## What NOT to Test

- Framework internals (SQLAlchemy, FastAPI, Pydantic)
- Implementation details that can change without observable behavior change
- Third-party library APIs you do not own

## Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/unit/platform/test_knowledge_retriever.py

# Verbose with coverage
pytest --tb=short -q
```
