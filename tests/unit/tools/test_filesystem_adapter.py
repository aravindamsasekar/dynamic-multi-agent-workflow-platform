"""Unit tests for FilesystemAdapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from platform.core.models.tool import ToolCall, ToolResult
from platform.tools.filesystem_adapter import FilesystemAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call(path: str, tool_use_id: str = "tu_1") -> ToolCall:
    return ToolCall(
        tool_use_id=tool_use_id,
        tool_name="filesystem_read_file",
        input={"path": path},
    )


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("Hello, World!", encoding="utf-8")
    (tmp_path / "README.md").write_text("# README\n\nThis is a test.", encoding="utf-8")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested content", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Successful reads
# ---------------------------------------------------------------------------


class TestFilesystemAdapterRead:
    async def test_reads_existing_text_file(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("hello.txt"))
        assert result.is_error is False
        assert result.content == "Hello, World!"

    async def test_reads_markdown_file(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("README.md"))
        assert result.is_error is False
        assert "# README" in result.content

    async def test_reads_nested_file_via_relative_path(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("subdir/nested.txt"))
        assert result.is_error is False
        assert result.content == "nested content"

    async def test_returns_correct_tool_use_id_on_success(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("hello.txt", tool_use_id="tu_abc"))
        assert result.tool_use_id == "tu_abc"

    async def test_result_is_tool_result_instance(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("hello.txt"))
        assert isinstance(result, ToolResult)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestFilesystemAdapterErrors:
    async def test_file_not_found_is_error(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("does_not_exist.txt"))
        assert result.is_error is True

    async def test_file_not_found_message_mentions_path(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("missing.txt"))
        assert "missing.txt" in result.content

    async def test_missing_path_parameter_is_error(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        call = ToolCall(
            tool_use_id="tu_1", tool_name="filesystem_read_file", input={}
        )
        result = await adapter.execute(call)
        assert result.is_error is True
        assert "path" in result.content.lower()

    async def test_empty_path_parameter_is_error(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call(""))
        assert result.is_error is True

    async def test_error_result_has_tool_use_id(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("missing.txt", tool_use_id="tu_err"))
        assert result.tool_use_id == "tu_err"


# ---------------------------------------------------------------------------
# Base directory configuration
# ---------------------------------------------------------------------------


class TestFilesystemAdapterBaseDir:
    async def test_default_base_dir_is_cwd(self) -> None:
        adapter = FilesystemAdapter()
        assert adapter._base_dir == Path(os.getcwd())

    async def test_base_dir_as_string(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=str(tmp_dir))
        result = await adapter.execute(_call("hello.txt"))
        assert result.is_error is False
        assert result.content == "Hello, World!"

    async def test_base_dir_as_path_object(self, tmp_dir: Path) -> None:
        adapter = FilesystemAdapter(base_dir=tmp_dir)
        result = await adapter.execute(_call("hello.txt"))
        assert result.is_error is False

    async def test_none_base_dir_resolves_to_cwd(self) -> None:
        adapter = FilesystemAdapter(base_dir=None)
        assert adapter._base_dir == Path.cwd()
