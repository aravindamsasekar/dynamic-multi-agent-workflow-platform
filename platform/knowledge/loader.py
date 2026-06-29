"""DocumentLoader — loads documents from disk into Document objects."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_TEXT_SUFFIXES: frozenset[str] = frozenset({
    ".md", ".txt", ".py", ".rst",
    ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".json", ".xml", ".html", ".css",
    ".js", ".ts", ".go", ".java", ".c", ".cpp", ".h", ".sh",
})


@dataclass
class Document:
    """A single document loaded from disk, ready for chunking."""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)


class DocumentLoader:
    """Loads supported documents from a directory or individual file.

    Supported formats: all plain-text extensions plus .pdf (requires pypdf).
    Unsupported extensions and empty files return None.
    """

    def load_collection(self, path: Path, collection_name: str) -> list[Document]:
        """Recursively load all supported documents from a directory."""
        docs: list[Document] = []
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                doc = self._load_file(file_path, collection_name)
                if doc is not None:
                    docs.append(doc)
        return docs

    def load_file(self, path: Path, collection_name: str) -> Document | None:
        """Load a single file. Returns None if unsupported or empty."""
        return self._load_file(path, collection_name)

    def _load_file(self, path: Path, collection_name: str) -> Document | None:
        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            return self._load_text(path, collection_name)
        if suffix == ".pdf":
            return self._load_pdf(path, collection_name)
        return None

    def _load_text(self, path: Path, collection_name: str) -> Document | None:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if not content.strip():
            return None
        return Document(
            content=content,
            metadata={
                "source_file": str(path),
                "file_type": path.suffix.lstrip(".").lower(),
                "collection": collection_name,
            },
        )

    def _load_pdf(self, path: Path, collection_name: str) -> Document | None:
        try:
            import pypdf  # optional dependency
        except ImportError:
            print(
                f"[knowledge] Skipping {path.name}: pypdf not installed. "
                "Run: pip install pypdf",
                file=sys.stderr,
            )
            return None
        try:
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
        except Exception as exc:
            print(f"[knowledge] Skipping {path.name}: {exc}", file=sys.stderr)
            return None
        if not text.strip():
            return None
        return Document(
            content=text,
            metadata={
                "source_file": str(path),
                "file_type": "pdf",
                "collection": collection_name,
            },
        )
