"""Unit tests for IEmbedder, MockEmbedder, and OpenAIEmbedder."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from platform.knowledge.embedder import IEmbedder, MockEmbedder, OpenAIEmbedder


def is_unit_vector(vec: list[float], tol: float = 1e-5) -> bool:
    return abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < tol


# ---------------------------------------------------------------------------
# IEmbedder is abstract
# ---------------------------------------------------------------------------


class TestIEmbedderAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            IEmbedder()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------


class TestMockEmbedder:
    async def test_embed_returns_one_vector_per_text(self):
        embedder = MockEmbedder(dimensions=8)
        results = await embedder.embed(["hello", "world", "test"])
        assert len(results) == 3

    async def test_embed_empty_list_returns_empty(self):
        embedder = MockEmbedder(dimensions=4)
        assert await embedder.embed([]) == []

    async def test_embed_single_text_returns_one_vector(self):
        embedder = MockEmbedder(dimensions=4)
        results = await embedder.embed(["single"])
        assert len(results) == 1

    async def test_vector_length_matches_dimensions(self):
        for dims in (4, 16, 64, 128):
            embedder = MockEmbedder(dimensions=dims)
            results = await embedder.embed(["text"])
            assert len(results[0]) == dims

    async def test_vectors_are_unit_normalised(self):
        embedder = MockEmbedder(dimensions=16)
        texts = ["hello", "world", "foo bar baz", ""]
        results = await embedder.embed(texts)
        for vec in results:
            assert is_unit_vector(vec), f"Vector not unit-length: norm={math.sqrt(sum(x*x for x in vec))}"

    async def test_same_text_always_returns_same_vector(self):
        embedder = MockEmbedder(dimensions=8)
        v1 = await embedder.embed(["deterministic text"])
        v2 = await embedder.embed(["deterministic text"])
        assert v1 == v2

    async def test_different_texts_return_different_vectors(self):
        embedder = MockEmbedder(dimensions=8)
        results = await embedder.embed(["hello", "world"])
        assert results[0] != results[1]

    async def test_order_of_texts_preserved(self):
        embedder = MockEmbedder(dimensions=4)
        solo_a = (await embedder.embed(["alpha"]))[0]
        solo_b = (await embedder.embed(["beta"]))[0]
        batch = await embedder.embed(["alpha", "beta"])
        assert batch[0] == solo_a
        assert batch[1] == solo_b

    async def test_repeated_text_in_batch_produces_identical_vectors(self):
        embedder = MockEmbedder(dimensions=4)
        results = await embedder.embed(["repeat", "repeat", "repeat"])
        assert results[0] == results[1] == results[2]

    async def test_default_dimensions_is_four(self):
        embedder = MockEmbedder()
        results = await embedder.embed(["text"])
        assert len(results[0]) == 4


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


def _make_mock_openai_client(embeddings: list[list[float]]) -> MagicMock:
    """Build a mock AsyncOpenAI client that returns the given embeddings."""
    mock_data = [MagicMock(embedding=vec) for vec in embeddings]
    mock_response = MagicMock()
    mock_response.data = mock_data

    mock_client = MagicMock()
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    return mock_client


class TestOpenAIEmbedder:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbedder(model="text-embedding-3-small")

    def test_accepts_explicit_api_key(self):
        client = _make_mock_openai_client([[0.1, 0.2]])
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key", _client=client)
        assert embedder._model == "text-embedding-3-small"

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        client = _make_mock_openai_client([[0.1]])
        embedder = OpenAIEmbedder(_client=client)
        assert embedder is not None

    async def test_embed_empty_list_returns_empty(self):
        client = _make_mock_openai_client([])
        embedder = OpenAIEmbedder(api_key="k", _client=client)
        result = await embedder.embed([])
        assert result == []
        client.embeddings.create.assert_not_called()

    async def test_embed_calls_openai_with_texts(self):
        raw = [[0.3, 0.4, 0.0]]
        client = _make_mock_openai_client(raw)
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="k", _client=client)
        await embedder.embed(["hello world"])

        client.embeddings.create.assert_called_once()
        _, kwargs = client.embeddings.create.call_args
        assert kwargs["input"] == ["hello world"]
        assert kwargs["model"] == "text-embedding-3-small"

    async def test_embed_returns_one_vector_per_text(self):
        raw = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        client = _make_mock_openai_client(raw)
        embedder = OpenAIEmbedder(api_key="k", _client=client)
        results = await embedder.embed(["a", "b", "c"])
        assert len(results) == 3

    async def test_embed_returns_unit_normalised_vectors(self):
        raw = [[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]]
        client = _make_mock_openai_client(raw)
        embedder = OpenAIEmbedder(api_key="k", _client=client)
        results = await embedder.embed(["a", "b", "c"])
        for vec in results:
            assert is_unit_vector(vec), f"Not unit-length: {vec}"

    async def test_known_vector_normalised_correctly(self):
        # [3, 4] has norm 5 → normalised to [0.6, 0.8]
        client = _make_mock_openai_client([[3.0, 4.0]])
        embedder = OpenAIEmbedder(api_key="k", _client=client)
        results = await embedder.embed(["text"])
        assert abs(results[0][0] - 0.6) < 1e-5
        assert abs(results[0][1] - 0.8) < 1e-5

    async def test_inject_client_bypasses_api_key_requirement(self):
        client = _make_mock_openai_client([[0.1, 0.9]])
        embedder = OpenAIEmbedder(_client=client)
        result = await embedder.embed(["text"])
        assert len(result) == 1
