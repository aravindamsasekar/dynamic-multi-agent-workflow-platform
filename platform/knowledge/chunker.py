"""TextChunker — splits Documents into overlapping Chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from platform.knowledge.loader import Document


@dataclass
class Chunk:
    """A text chunk produced from a Document, ready for embedding."""

    content: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)


class TextChunker:
    """Splits document text into fixed-size overlapping chunks.

    When a natural boundary (paragraph break > newline > space) falls within the
    size window, the split is placed there rather than at the hard character limit.
    The overlap region at the end of chunk N appears again at the start of chunk N+1.
    """

    def __init__(self, size: int = 1000, overlap: int = 200) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= size:
            raise ValueError(f"overlap ({overlap}) must be less than size ({size})")
        self._size = size
        self._overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into overlapping chunks, preserving document metadata."""
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self._size, len(text))

            if end < len(text):
                # Prefer natural boundaries: paragraph > newline > space
                for sep in ("\n\n", "\n", " "):
                    pos = text.rfind(sep, start + 1, end)
                    if pos > start:
                        end = pos + len(sep)
                        break

            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_index=idx,
                        metadata={**document.metadata, "chunk_index": str(idx)},
                    )
                )
                idx += 1

            # All text has been consumed — stop regardless of overlap setting
            if end >= len(text):
                break

            next_start = end - self._overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks
