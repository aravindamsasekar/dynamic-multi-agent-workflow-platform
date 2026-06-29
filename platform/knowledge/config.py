"""KnowledgeConfig and CollectionConfig — configuration dataclasses for the knowledge layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CollectionConfig:
    """Configuration for a single named knowledge collection."""

    name: str
    path: str


@dataclass
class KnowledgeConfig:
    """Top-level configuration for the knowledge layer.

    Parsed from knowledge_config.yaml via from_dict().
    """

    embedding_model: str = "text-embedding-3-small"
    vector_store_path: str = "data/knowledge"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    collections: list[CollectionConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeConfig:
        """Parse from the knowledge_config.yaml structure.

        Accepts either a top-level dict with a "knowledge" key, or the inner
        section directly.
        """
        section: dict[str, Any] = data.get("knowledge", data)
        embedding: dict[str, Any] = section.get("embedding", {})
        vector_store: dict[str, Any] = section.get("vector_store", {})
        chunking: dict[str, Any] = section.get("chunking", {})
        retrieval: dict[str, Any] = section.get("retrieval", {})
        collections = [
            CollectionConfig(name=c["name"], path=c["path"])
            for c in section.get("collections", [])
        ]
        return cls(
            embedding_model=embedding.get("model", "text-embedding-3-small"),
            vector_store_path=vector_store.get("path", "data/knowledge"),
            chunk_size=chunking.get("size", 1000),
            chunk_overlap=chunking.get("overlap", 200),
            top_k=retrieval.get("top_k", 5),
            collections=collections,
        )
