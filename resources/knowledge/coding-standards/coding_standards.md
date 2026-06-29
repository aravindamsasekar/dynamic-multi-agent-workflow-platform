# Coding Standards

## Python

### Naming

- Modules, variables, functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore` (one underscore, not two)
- Type aliases: `PascalCase` (e.g., `SessionFactory`)

### Type Annotations

- All function signatures must include type annotations for parameters and return types.
- Use `from __future__ import annotations` at the top of every module to enable deferred evaluation.
- Prefer `X | None` over `Optional[X]` (Python 3.10+ union syntax).
- Use `list[T]` and `dict[K, V]` (lowercase) not `List[T]` and `Dict[K, V]` from `typing`.

### Imports

- Group imports: standard library → third-party → internal platform.
- Always use absolute imports within the platform (`from platform.core.models.tool import ToolCall`, not relative).
- Do not use wildcard imports (`from module import *`).

### Functions and Methods

- Functions should do one thing. If a function needs a long comment to explain what it does, split it.
- Keep function length under 50 lines. Longer functions are a code smell.
- Avoid positional-only arguments in public APIs. Prefer keyword arguments for clarity at call sites.
- Avoid mutable default arguments (`def f(items=[])` is a bug, use `def f(items=None)` and assign inside).

### Classes

- Prefer dataclasses or Pydantic models over raw dicts for structured data.
- Avoid inheriting from multiple concrete classes (multiple inheritance from ABCs is acceptable).
- `__init__` should not do significant work. Heavy initialization belongs in a factory or async `setup()` method.

### Error Handling

- Catch specific exceptions, not bare `except:` or `except Exception:` (except at top-level boundary handlers).
- Include context in error messages: `raise ValueError(f"Expected positive integer, got {value!r}")`.
- Use custom exception classes from `platform.core.exceptions` for platform-level failures.
- Do not swallow exceptions silently. Log or re-raise.

### Async

- Any function that calls `await` must be declared `async def`.
- Do not mix sync and async I/O on the same thread. Use `asyncio.run()` only at the top-most entry point.
- Long-running CPU work inside an async function should be delegated to a thread pool via `asyncio.to_thread()`.

## Formatting and Linting

- Code is formatted with `black` (line length 100).
- Import order is enforced by `isort` with `profile = "black"`.
- All public symbols are type-checked with `mypy --strict`.
- Lint with `ruff` — zero warnings is the target.

## Commits and PRs

- Commit messages use the imperative mood: "Add feature" not "Added feature".
- Each commit should pass CI independently (no "WIP" commits on main).
- PR title summarises the change in under 70 characters.
- PR description must link to the relevant ticket and explain WHY the change is being made.
