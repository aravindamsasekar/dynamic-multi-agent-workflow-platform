"""IEmbedder interface, OpenAIEmbedder, and MockEmbedder."""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod

from openai import AsyncOpenAI


class IEmbedder(ABC):
    """Abstract interface for text embedding models.

    All implementations return L2-normalised float vectors suitable for
    cosine similarity with FAISSVectorStore (IndexFlatIP).
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one unit-normalised vector per text."""
        ...


class OpenAIEmbedder(IEmbedder):
    """Embeds texts via the OpenAI Embeddings API.

    API key resolution: constructor arg → OPENAI_API_KEY env var.
    Inject _client for testing without network calls.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        _client: AsyncOpenAI | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key and _client is None:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Provide it via api_key or OPENAI_API_KEY env var."
            )
        self._model = model
        self._client = _client or AsyncOpenAI(api_key=resolved_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [_l2_normalize(item.embedding) for item in response.data]


class MockEmbedder(IEmbedder):
    """Deterministic embedder for testing — no network calls.

    Maps each text to a unit-normalised vector derived from the text's
    SHA-256 digest. Same input always produces the same output.
    """

    def __init__(self, dimensions: int = 4) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [
            (digest[i % len(digest)] / 127.5) - 1.0
            for i in range(self._dimensions)
        ]
        return _l2_normalize(raw)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        d = len(vec)
        return [1.0 / math.sqrt(d)] * d
    return [x / norm for x in vec]
