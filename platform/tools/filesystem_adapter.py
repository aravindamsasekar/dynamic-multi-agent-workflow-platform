"""FilesystemAdapter — reads local files for use in generated agent workflows."""

from __future__ import annotations

from pathlib import Path

from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.tool import ToolCall, ToolResult


class FilesystemAdapter(IToolAdapter):
    """Reads a local file and returns its UTF-8 contents.

    Configured with an optional base_dir (defaults to CWD).
    The tool call input must contain a "path" key with a relative or absolute path.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    async def execute(self, call: ToolCall) -> ToolResult:
        file_path = call.input.get("path", "")
        if not file_path:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content="Error: 'path' parameter is required",
                is_error=True,
            )
        try:
            full_path = (self._base_dir / file_path).resolve()
            content = full_path.read_text(encoding="utf-8")
            return ToolResult(tool_use_id=call.tool_use_id, content=content, is_error=False)
        except FileNotFoundError:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=f"File not found: {file_path}",
                is_error=True,
            )
        except PermissionError:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=f"Permission denied: {file_path}",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                content=f"Error reading file: {exc}",
                is_error=True,
            )
