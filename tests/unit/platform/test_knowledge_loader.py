"""Unit tests for DocumentLoader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from platform.knowledge.loader import Document, DocumentLoader


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------


class TestDocument:
    def test_stores_content_and_metadata(self):
        doc = Document(content="Hello world", metadata={"source_file": "test.md"})
        assert doc.content == "Hello world"
        assert doc.metadata["source_file"] == "test.md"

    def test_default_metadata_is_empty(self):
        doc = Document(content="text")
        assert doc.metadata == {}

    def test_equality(self):
        a = Document(content="x", metadata={"k": "v"})
        b = Document(content="x", metadata={"k": "v"})
        assert a == b


# ---------------------------------------------------------------------------
# Load single file
# ---------------------------------------------------------------------------


class TestDocumentLoaderFile:
    def test_loads_markdown_file(self, tmp_path: Path):
        f = tmp_path / "guide.md"
        f.write_text("# Title\n\nSome content.", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "test-collection")
        assert doc is not None
        assert "Title" in doc.content
        assert doc.metadata["file_type"] == "md"
        assert doc.metadata["collection"] == "test-collection"
        assert doc.metadata["source_file"] == str(f)

    def test_loads_text_file(self, tmp_path: Path):
        f = tmp_path / "notes.txt"
        f.write_text("Plain text content.", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "notes")
        assert doc is not None
        assert doc.content == "Plain text content."
        assert doc.metadata["file_type"] == "txt"

    def test_loads_python_file(self, tmp_path: Path):
        f = tmp_path / "module.py"
        f.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "source")
        assert doc is not None
        assert "def hello" in doc.content
        assert doc.metadata["file_type"] == "py"

    def test_loads_yaml_file(self, tmp_path: Path):
        f = tmp_path / "config.yaml"
        f.write_text("key: value\nother: 42\n", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "col")
        assert doc is not None
        assert "key: value" in doc.content
        assert doc.metadata["file_type"] == "yaml"

    def test_loads_rst_file(self, tmp_path: Path):
        f = tmp_path / "readme.rst"
        f.write_text("Title\n=====\n\nContent here.", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "col")
        assert doc is not None
        assert doc.metadata["file_type"] == "rst"

    def test_loads_json_file(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        doc = DocumentLoader().load_file(f, "col")
        assert doc is not None

    def test_unsupported_extension_returns_none(self, tmp_path: Path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n")
        assert DocumentLoader().load_file(f, "col") is None

    def test_binary_extension_returns_none(self, tmp_path: Path):
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK\x03\x04")
        assert DocumentLoader().load_file(f, "col") is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        assert DocumentLoader().load_file(f, "col") is None

    def test_whitespace_only_file_returns_none(self, tmp_path: Path):
        f = tmp_path / "blank.txt"
        f.write_text("   \n\n   \t  ", encoding="utf-8")
        assert DocumentLoader().load_file(f, "col") is None

    def test_metadata_includes_collection_name(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("Content.", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "my-collection")
        assert doc is not None
        assert doc.metadata["collection"] == "my-collection"

    def test_metadata_includes_source_file_path(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("Content.", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "col")
        assert doc is not None
        assert doc.metadata["source_file"] == str(f)

    def test_extension_case_insensitive(self, tmp_path: Path):
        f = tmp_path / "README.MD"
        f.write_text("Content.", encoding="utf-8")
        doc = DocumentLoader().load_file(f, "col")
        assert doc is not None
        assert doc.metadata["file_type"] == "md"

    def test_pdf_skipped_when_pypdf_unavailable(self, tmp_path: Path, capsys):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")

        with patch.dict("sys.modules", {"pypdf": None}):
            result = DocumentLoader().load_file(f, "col")

        assert result is None
        captured = capsys.readouterr()
        assert "pypdf" in captured.err

    def test_content_preserved_exactly(self, tmp_path: Path):
        text = "Line one.\nLine two.\n\nParagraph two."
        f = tmp_path / "doc.txt"
        f.write_text(text, encoding="utf-8")
        doc = DocumentLoader().load_file(f, "col")
        assert doc is not None
        assert doc.content == text


# ---------------------------------------------------------------------------
# Load collection (directory)
# ---------------------------------------------------------------------------


class TestDocumentLoaderCollection:
    def test_loads_all_supported_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("Markdown doc.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Text doc.", encoding="utf-8")
        (tmp_path / "c.py").write_text("# Python", encoding="utf-8")
        docs = DocumentLoader().load_collection(tmp_path, "test")
        assert len(docs) == 3

    def test_skips_unsupported_files(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text("Good.", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        docs = DocumentLoader().load_collection(tmp_path, "test")
        assert len(docs) == 1

    def test_skips_empty_files(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text("Content.", encoding="utf-8")
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        docs = DocumentLoader().load_collection(tmp_path, "test")
        assert len(docs) == 1

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        assert DocumentLoader().load_collection(tmp_path, "test") == []

    def test_loads_files_in_subdirectories(self, tmp_path: Path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("Nested content.", encoding="utf-8")
        docs = DocumentLoader().load_collection(tmp_path, "test")
        assert len(docs) == 1
        assert "Nested content" in docs[0].content

    def test_loads_files_in_deep_subdirectories(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.md").write_text("Deep content.", encoding="utf-8")
        docs = DocumentLoader().load_collection(tmp_path, "test")
        assert len(docs) == 1

    def test_collection_name_set_on_all_docs(self, tmp_path: Path):
        for name in ("a.md", "b.txt", "c.py"):
            (tmp_path / name).write_text("Content.", encoding="utf-8")
        docs = DocumentLoader().load_collection(tmp_path, "my-col")
        assert all(d.metadata["collection"] == "my-col" for d in docs)

    def test_ignores_directories_themselves(self, tmp_path: Path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "doc.md").write_text("Content.", encoding="utf-8")
        docs = DocumentLoader().load_collection(tmp_path, "test")
        assert len(docs) == 1

    def test_multiple_file_types_all_loaded(self, tmp_path: Path):
        files = {
            "guide.md": "Guide content.",
            "notes.txt": "Notes content.",
            "script.py": "print('hi')",
            "config.yaml": "key: value",
        }
        for name, content in files.items():
            (tmp_path / name).write_text(content, encoding="utf-8")
        docs = DocumentLoader().load_collection(tmp_path, "mixed")
        assert len(docs) == 4
        types = {d.metadata["file_type"] for d in docs}
        assert types == {"md", "txt", "py", "yaml"}
