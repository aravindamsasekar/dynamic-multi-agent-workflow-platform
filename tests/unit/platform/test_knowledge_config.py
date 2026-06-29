"""Unit tests for KnowledgeConfig and CollectionConfig."""

from __future__ import annotations

from platform.knowledge.config import CollectionConfig, KnowledgeConfig


class TestCollectionConfig:
    def test_stores_name_and_path(self):
        c = CollectionConfig(name="coding-standards", path="resources/knowledge/coding-standards")
        assert c.name == "coding-standards"
        assert c.path == "resources/knowledge/coding-standards"

    def test_equality(self):
        a = CollectionConfig(name="docs", path="p/docs")
        b = CollectionConfig(name="docs", path="p/docs")
        assert a == b

    def test_inequality_on_name(self):
        a = CollectionConfig(name="a", path="p")
        b = CollectionConfig(name="b", path="p")
        assert a != b


class TestKnowledgeConfigDefaults:
    def test_default_embedding_model(self):
        assert KnowledgeConfig().embedding_model == "text-embedding-3-small"

    def test_default_vector_store_path(self):
        assert KnowledgeConfig().vector_store_path == "data/knowledge"

    def test_default_chunk_size(self):
        assert KnowledgeConfig().chunk_size == 1000

    def test_default_chunk_overlap(self):
        assert KnowledgeConfig().chunk_overlap == 200

    def test_default_top_k(self):
        assert KnowledgeConfig().top_k == 5

    def test_default_collections_empty(self):
        assert KnowledgeConfig().collections == []

    def test_from_dict_empty_section_uses_defaults(self):
        cfg = KnowledgeConfig.from_dict({"knowledge": {}})
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.vector_store_path == "data/knowledge"
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.top_k == 5
        assert cfg.collections == []


class TestKnowledgeConfigFromDict:
    def test_parses_full_config(self):
        data = {
            "knowledge": {
                "embedding": {"model": "text-embedding-3-large"},
                "vector_store": {"path": "data/custom"},
                "chunking": {"size": 500, "overlap": 100},
                "retrieval": {"top_k": 3},
                "collections": [
                    {"name": "docs", "path": "resources/knowledge/docs"},
                    {"name": "runbooks", "path": "resources/knowledge/runbooks"},
                ],
            }
        }
        cfg = KnowledgeConfig.from_dict(data)
        assert cfg.embedding_model == "text-embedding-3-large"
        assert cfg.vector_store_path == "data/custom"
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 100
        assert cfg.top_k == 3
        assert len(cfg.collections) == 2
        assert cfg.collections[0].name == "docs"
        assert cfg.collections[0].path == "resources/knowledge/docs"
        assert cfg.collections[1].name == "runbooks"

    def test_partial_config_keeps_other_defaults(self):
        data = {"knowledge": {"embedding": {"model": "text-embedding-3-large"}}}
        cfg = KnowledgeConfig.from_dict(data)
        assert cfg.embedding_model == "text-embedding-3-large"
        assert cfg.chunk_size == 1000

    def test_accepts_dict_without_knowledge_wrapper(self):
        data = {
            "embedding": {"model": "text-embedding-3-small"},
            "collections": [],
        }
        cfg = KnowledgeConfig.from_dict(data)
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_empty_collections_list(self):
        cfg = KnowledgeConfig.from_dict({"knowledge": {"collections": []}})
        assert cfg.collections == []

    def test_collections_preserve_order(self):
        data = {
            "knowledge": {
                "collections": [
                    {"name": "a", "path": "p/a"},
                    {"name": "b", "path": "p/b"},
                    {"name": "c", "path": "p/c"},
                ]
            }
        }
        cfg = KnowledgeConfig.from_dict(data)
        assert [c.name for c in cfg.collections] == ["a", "b", "c"]

    def test_missing_embedding_section_uses_default_model(self):
        cfg = KnowledgeConfig.from_dict({"knowledge": {"chunking": {"size": 500}}})
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_missing_chunking_section_uses_defaults(self):
        cfg = KnowledgeConfig.from_dict({"knowledge": {}})
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200

    def test_missing_retrieval_section_uses_default_top_k(self):
        cfg = KnowledgeConfig.from_dict({"knowledge": {}})
        assert cfg.top_k == 5
