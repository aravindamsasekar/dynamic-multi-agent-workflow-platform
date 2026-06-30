"""PatternSelector — deterministic pattern selection for V3.1."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import GoalAnalysis, TaskType


class PatternSelector:
    """Selects an execution pattern for a given GoalAnalysis.

    V3.1: CODE_REVIEW → parallel_specialist (from registry default).
    All other task types return an empty string; the validator catches this.
    """

    def select(self, analysis: GoalAnalysis, registry: CapabilityRegistry) -> str:
        """Return the pattern string for this goal, or '' if unsupported."""
        if analysis.task_type == TaskType.UNSUPPORTED:
            return ""
        pattern_desc = registry.get_default_pattern_for_task_type(analysis.task_type.value)
        if pattern_desc is None:
            return ""
        return pattern_desc.pattern
