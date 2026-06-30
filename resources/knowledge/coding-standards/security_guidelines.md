# Security Review Guidelines

## Overview

Every pull request must be reviewed for security implications before merging.
These guidelines define what to look for, how to classify severity, and what
evidence to include in a security assessment.

---

## 1. Input Validation

All inputs from external sources (HTTP requests, environment variables, file paths,
user-provided data) must be validated before use.

**Required checks:**
- Validate type, format, and length at system boundaries
- Reject or sanitize before passing to downstream systems
- Never trust caller-provided identifiers for access control decisions

**Red flags in diff:**
- Data read from `request.json()`, `os.environ`, or file I/O without validation
- String interpolation of user input into shell commands, SQL, or template strings
- Missing length/type checks on external parameters

**Severity:** High if unvalidated input reaches a privileged operation; Medium otherwise.

---

## 2. SQL and Query Safety

**Required:**
- Use parameterized queries or ORM query builders exclusively
- Never concatenate user input into raw SQL strings
- Validate identifiers (table/column names) against an allowlist if dynamic queries are unavoidable

**Red flags:**
- `f"SELECT ... WHERE id = {user_id}"` or `cursor.execute("... " + value)`
- Dynamic table/column names derived from request parameters
- Raw `.execute()` calls with string-formatted queries

**Severity:** Critical — SQL injection is directly exploitable.

---

## 3. Secrets Management

Secrets must never appear in source code, logs, or version-controlled files.

**Required:**
- Read secrets exclusively from environment variables or a secrets manager
- Use placeholder names (`SECRET_KEY`, `API_TOKEN`) never literal values
- Rotate secrets immediately if accidentally committed

**Red flags in diff:**
- String literals that look like tokens, passwords, or API keys (long random strings, `sk-`, `ghp_`, `AKIA`)
- Secrets passed as function arguments inline
- Logging statements that include authentication headers or credentials

**Severity:** Critical if an actual secret value is present; High for patterns that suggest future leakage.

---

## 4. Authentication and Authorization

**Required:**
- Verify identity before returning any user-specific data
- Enforce authorization on every operation, not just at the entry point
- Use existing auth middleware; do not implement custom auth logic without review

**Red flags:**
- Endpoints that skip the standard auth decorator/middleware
- Object-level authorization missing (fetching by ID without verifying ownership)
- Privilege escalation paths (users modifying their own roles or permissions)
- Hard-coded bypass conditions (`if user_id == "admin"`)

**Severity:** Critical for missing auth on sensitive operations; High for privilege escalation risks.

---

## 5. Dependency and Supply Chain Review

**Required:**
- New dependencies must be vetted for known CVEs
- Pin versions in `requirements.txt` or `pyproject.toml`; do not use open ranges for production dependencies
- Avoid unmaintained packages (last release > 2 years, no active maintainer)

**Red flags:**
- `pip install package` without version pin
- New `import` of unfamiliar third-party library without corresponding lockfile update
- Version ranges that allow major updates (`>=1.0`)

**Severity:** High if CVE exists; Medium for unpinned production dependencies.

---

## 6. Error Handling and Information Disclosure

**Required:**
- Return generic error messages to clients; log detailed errors server-side only
- Do not expose stack traces, internal paths, or system information in API responses
- Handle all exceptions — unhandled exceptions can leak information or crash services

**Red flags:**
- `except Exception as e: return {"error": str(e)}` in API handlers
- Tracebacks serialized into HTTP responses
- Debug mode left enabled in production paths

**Severity:** Medium for information disclosure; High if combined with sensitive data exposure.

---

## 7. Severity Classification

| Severity | Definition | Required Action |
|----------|-----------|----------------|
| Critical | Direct exploitability; secret leakage; auth bypass | Block merge; immediate fix required |
| High     | Likely exploitable with moderate effort; insecure patterns | Block merge; fix before approval |
| Medium   | Exploitable under specific conditions; bad practices | Fix required or document accepted risk |
| Low      | Defense-in-depth improvements; minor concerns | Fix encouraged; not a blocker |
| Info     | Observations and suggestions | Optional improvement |

A PR with any Critical or High finding must request changes. Medium findings require
explicit acknowledgement. Low and Info findings may be approved at reviewer discretion.

---

## 8. Evidence Standards

Every security finding must include:
1. **Location** — file name and line number from the diff
2. **Pattern** — what specific code construct was flagged
3. **Risk** — what attack or failure mode this enables
4. **Recommendation** — the specific change required to remediate

Vague findings ("this could be insecure") without specific evidence are not acceptable.
