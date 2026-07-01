"""ConsoleObserver — emits structured workflow events to stdout."""

from __future__ import annotations

import sys

from platform.core.interfaces.observer import IObserver
from platform.core.models.events import WorkflowEvent


class ConsoleObserver(IObserver):
    """Writes workflow events as JSON lines to stdout.

    Default IObserver implementation for V1. Suitable for local development
    and demo runs. Replace with a file or external sink observer for production.
    """

    def on_event(self, event: WorkflowEvent) -> None:
        line = event.model_dump_json() + "\n"
        # On Windows, stdout.encoding may be cp1252 which can't represent Unicode.
        # Write to the binary buffer (available on real file objects) to avoid
        # UnicodeEncodeError when file content or LLM output contains non-ASCII chars.
        buf = getattr(sys.stdout, "buffer", None)
        if buf is not None:
            buf.write(line.encode("utf-8", errors="replace"))
            buf.flush()
        else:
            sys.stdout.write(line)
