"""Unit tests for TextChunker and Chunk."""

from __future__ import annotations

import pytest

from platform.knowledge.chunker import Chunk, TextChunker
from platform.knowledge.loader import Document


def make_doc(content: str, **meta: str) -> Document:
    return Document(content=content, metadata={"source_file": "test.md", **meta})


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


class TestChunk:
    def test_stores_content_index_metadata(self):
        chunk = Chunk(content="text", chunk_index=0, metadata={"source_file": "f.md"})
        assert chunk.content == "text"
        assert chunk.chunk_index == 0
        assert chunk.metadata["source_file"] == "f.md"

    def test_default_metadata_is_empty(self):
        chunk = Chunk(content="text", chunk_index=0)
        assert chunk.metadata == {}


# ---------------------------------------------------------------------------
# TextChunker construction validation
# ---------------------------------------------------------------------------


class TestTextChunkerInit:
    def test_default_size_and_overlap(self):
        chunker = TextChunker()
        assert chunker._size == 1000
        assert chunker._overlap == 200

    def test_custom_size_and_overlap(self):
        chunker = TextChunker(size=500, overlap=50)
        assert chunker._size == 500
        assert chunker._overlap == 50

    def test_zero_overlap_is_allowed(self):
        TextChunker(size=100, overlap=0)

    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="size must be positive"):
            TextChunker(size=0)

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="size must be positive"):
            TextChunker(size=-1)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            TextChunker(size=100, overlap=-1)

    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="overlap.*must be less than size"):
            TextChunker(size=100, overlap=100)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError, match="overlap.*must be less than size"):
            TextChunker(size=100, overlap=150)


# ---------------------------------------------------------------------------
# Empty / whitespace documents
# ---------------------------------------------------------------------------


class TestTextChunkerEmpty:
    def test_empty_content_returns_empty_list(self):
        assert TextChunker().chunk(make_doc("")) == []

    def test_whitespace_only_returns_empty_list(self):
        assert TextChunker().chunk(make_doc("   \n\n\t  ")) == []

    def test_single_newline_returns_empty_list(self):
        assert TextChunker().chunk(make_doc("\n")) == []


# ---------------------------------------------------------------------------
# Short documents (fit in one chunk)
# ---------------------------------------------------------------------------


class TestTextChunkerShortDocuments:
    def test_content_shorter_than_size_produces_one_chunk(self):
        doc = make_doc("Short text.")
        chunks = TextChunker(size=1000, overlap=200).chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].chunk_index == 0

    def test_content_exactly_size_produces_one_chunk(self):
        text = "a" * 1000
        chunks = TextChunker(size=1000, overlap=200).chunk(make_doc(text))
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_single_word_produces_one_chunk(self):
        chunks = TextChunker(size=100, overlap=10).chunk(make_doc("hello"))
        assert len(chunks) == 1
        assert chunks[0].content == "hello"


# ---------------------------------------------------------------------------
# Chunk indexing
# ---------------------------------------------------------------------------


class TestTextChunkerIndexing:
    def test_chunk_indices_are_zero_based_sequential(self):
        text = "word " * 500  # 2500 chars
        chunks = TextChunker(size=1000, overlap=200).chunk(make_doc(text))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_long_text_produces_multiple_chunks(self):
        text = "word " * 500
        chunks = TextChunker(size=1000, overlap=200).chunk(make_doc(text))
        assert len(chunks) >= 2

    def test_zero_overlap_chunk_count(self):
        text = "a" * 2000
        chunks = TextChunker(size=1000, overlap=0).chunk(make_doc(text))
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Content coverage and overlap
# ---------------------------------------------------------------------------


class TestTextChunkerContent:
    def test_first_chunk_starts_at_document_beginning(self):
        text = "START " + "word " * 300
        chunks = TextChunker(size=500, overlap=100).chunk(make_doc(text))
        assert chunks[0].content.startswith("START")

    def test_last_chunk_ends_at_document_end(self):
        text = "word " * 300 + "END"
        chunks = TextChunker(size=500, overlap=100).chunk(make_doc(text))
        assert chunks[-1].content.endswith("END")

    def test_adjacent_chunks_share_overlap_region(self):
        # No spaces/newlines → exact character splits → deterministic overlap positions
        text = "x" * 2000
        chunks = TextChunker(size=1000, overlap=200).chunk(make_doc(text))
        assert len(chunks) >= 2
        assert chunks[0].content[-200:] == chunks[1].content[:200]

    def test_zero_overlap_no_shared_content(self):
        # No natural split points → exact halves, no overlap
        text = "a" * 1000 + "b" * 1000
        chunks = TextChunker(size=1000, overlap=0).chunk(make_doc(text))
        assert len(chunks) == 2
        assert chunks[0].content == "a" * 1000
        assert chunks[1].content == "b" * 1000
        assert chunks[0].content + chunks[1].content == text

    def test_no_content_lost(self):
        # Every character in the source appears in at least one chunk
        text = "The quick brown fox " * 100  # 2000 chars
        chunks = TextChunker(size=600, overlap=100).chunk(make_doc(text))
        # First chunk starts from the beginning, last ends at the end
        assert text.startswith(chunks[0].content[:10])
        assert text.endswith(chunks[-1].content[-10:])


# ---------------------------------------------------------------------------
# Boundary preference
# ---------------------------------------------------------------------------


class TestTextChunkerBoundaries:
    def test_prefers_paragraph_break(self):
        # First paragraph (16 chars) + "\n\n" (2) = 18 chars, well within size=25
        text = "First paragraph." + "\n\n" + "Second paragraph."
        chunks = TextChunker(size=25, overlap=0).chunk(make_doc(text))
        assert chunks[0].content == "First paragraph.\n\n"

    def test_prefers_newline_over_space(self):
        # "\n" at index 8, space at index 4 — newline should win (tried first in rfind)
        text = "line one\nline two more words here"
        chunks = TextChunker(size=15, overlap=0).chunk(make_doc(text))
        assert chunks[0].content == "line one\n"

    def test_prefers_space_over_mid_character(self):
        # space at index 5 → split at 6 ("hello ")
        text = "hello world test"
        chunks = TextChunker(size=10, overlap=0).chunk(make_doc(text))
        assert chunks[0].content == "hello "

    def test_falls_back_to_hard_split_when_no_boundary(self):
        # All "x" — no natural boundaries → exact size split
        text = "x" * 20
        chunks = TextChunker(size=10, overlap=0).chunk(make_doc(text))
        assert chunks[0].content == "x" * 10
        assert chunks[1].content == "x" * 10


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


class TestTextChunkerMetadata:
    def test_document_metadata_propagated_to_all_chunks(self):
        text = "word " * 500
        doc = make_doc(text, collection="test-col", file_type="md")
        chunks = TextChunker(size=500, overlap=100).chunk(doc)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.metadata["collection"] == "test-col"
            assert chunk.metadata["file_type"] == "md"
            assert chunk.metadata["source_file"] == "test.md"

    def test_chunk_index_added_to_metadata(self):
        text = "word " * 500
        chunks = TextChunker(size=500, overlap=100).chunk(make_doc(text))
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == str(i)

    def test_source_file_propagated(self):
        doc = Document(
            content="Content line. " * 100,
            metadata={"source_file": "resources/knowledge/docs/guide.md"},
        )
        chunks = TextChunker(size=200, overlap=50).chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "resources/knowledge/docs/guide.md"

    def test_original_document_metadata_not_mutated(self):
        doc = make_doc("word " * 500)
        original_meta = dict(doc.metadata)
        TextChunker(size=500, overlap=100).chunk(doc)
        assert doc.metadata == original_meta
