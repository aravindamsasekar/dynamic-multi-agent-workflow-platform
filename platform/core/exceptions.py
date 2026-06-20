"""Platform-level exceptions."""


class WorkflowNotFound(Exception):
    """Raised when a workflow_id is not found in WorkflowRegistry."""


class AgentNotFound(Exception):
    """Raised when an agent_id is not found in AgentRegistry."""


class ToolNotFound(Exception):
    """Raised when a tool_name is not found in ToolRegistry."""


class PolicyViolation(Exception):
    """Raised when a policy rule blocks execution."""


class HITLRejected(Exception):
    """Raised when a human rejects a HITL approval request."""


class PatternExecutionError(Exception):
    """Raised when a pattern executor fails during execution."""


class ConfigValidationError(Exception):
    """Raised when a workflow YAML file fails schema validation."""
