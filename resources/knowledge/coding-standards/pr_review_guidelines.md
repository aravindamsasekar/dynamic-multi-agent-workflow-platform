# Pull Request Review Guidelines

## Purpose

These guidelines define what reviewers must check on every pull request. They ensure consistent, high-quality code lands in the main branch.

## Required Checks

### 1. Scope and Size

- A PR should address a single concern. Mixed concerns (feature + refactor + bugfix) are grounds for a "Request Changes" verdict asking the author to split the PR.
- PRs larger than 400 changed lines require justification in the description. If none is given, request one before approving.
- Trivial churn (whitespace, cosmetic renames not tied to a ticket) should be called out.

### 2. Tests

- Every non-trivial change to business logic must include tests. Missing tests are a blocker.
- Tests must be deterministic. Flaky tests that pass probabilistically are not acceptable.
- Test coverage should reflect the risk surface of the change: high-risk paths need unit AND integration tests.
- Check that tests actually exercise the changed code path (not just existing paths that happen to pass).

### 3. Error Handling

- New API endpoints must handle and return structured errors, not raw exceptions.
- Database operations must be wrapped in transactions where atomicity matters.
- External calls (HTTP, queue, file I/O) must handle transient failures gracefully (retry, timeout, circuit breaker).

### 4. Security

- No secrets, tokens, or credentials in code, comments, or test fixtures.
- User-supplied input must be validated before use. Reject early at system boundaries.
- SQL queries must use parameterized statements. String-concatenated queries are an automatic rejection.
- Dependencies added in this PR must be reviewed: check for known CVEs and ensure the license is compatible.

### 5. Documentation

- Public APIs (functions, classes, HTTP endpoints) visible to other teams need a docstring or schema comment.
- Complex algorithms or non-obvious design decisions require inline explanations of the WHY, not the WHAT.
- README or runbook updates are expected when the change affects operational behavior or deployment.

### 6. Performance

- Queries inside loops ("N+1 problem") must be refactored to batch operations.
- Hot paths (called on every request) must not introduce unbounded memory growth.
- New background jobs need an explanation of their expected runtime and resource usage.

## Verdict Guide

| Verdict | When to use |
|---|---|
| Approve | Code is correct, tests pass, no significant issues found |
| Request Changes | Issues that must be fixed before merging (missing tests, security flaw, correctness bug) |
| Comment | Minor suggestions that can be addressed at the author's discretion |
