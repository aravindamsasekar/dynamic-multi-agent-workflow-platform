"""Phase 5 manifest tests — github-integration and knowledge-search catalog entries."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform.extensions.catalog import ExtensionCatalog

_EXTENSIONS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "resources" / "extensions"
)

# Loaded once for the whole module — mirrors how api/dependencies.py uses it.
_CATALOG = ExtensionCatalog.load(_EXTENSIONS_DIR)

_GITHUB_CAPS = {
    "fetch_pr_data",
    "fetch_github_diff",
    "fetch_changed_files",
    "review_code_quality",
    "assess_architecture",
    "check_standards",
    "assess_security",
    "assess_testing",
    "assess_reliability",
    "synthesize_findings",
    "produce_final_report",
}

_GITHUB_TOOLS = {"github_get_pr", "github_get_files", "github_get_diff", "mcp_get_pr_comments"}

_KNOWLEDGE_CAPS = {"search_knowledge", "search_coding_standards", "search_architecture"}


# ---------------------------------------------------------------------------
# Catalog completeness
# ---------------------------------------------------------------------------


class TestCatalogCompleteness:
    def test_catalog_has_three_extensions(self) -> None:
        assert len(_CATALOG.all()) == 3

    def test_catalog_contains_filesystem_reader(self) -> None:
        assert _CATALOG.get("filesystem-reader") is not None

    def test_catalog_contains_github_integration(self) -> None:
        assert _CATALOG.get("github-integration") is not None

    def test_catalog_contains_knowledge_search(self) -> None:
        assert _CATALOG.get("knowledge-search") is not None


# ---------------------------------------------------------------------------
# github-integration manifest
# ---------------------------------------------------------------------------


class TestGithubIntegrationManifest:
    @pytest.fixture(autouse=True)
    def pkg(self):
        self._pkg = _CATALOG.get("github-integration")

    def test_id(self) -> None:
        assert self._pkg.id == "github-integration"

    def test_name(self) -> None:
        assert self._pkg.name == "GitHub Integration"

    def test_version(self) -> None:
        assert self._pkg.version == "1.0.0"

    def test_category(self) -> None:
        assert self._pkg.category == "github"

    def test_provides_static_agent(self) -> None:
        assert "static_agent" in self._pkg.provides

    def test_does_not_provide_runtime_agent(self) -> None:
        assert "runtime_agent" not in self._pkg.provides

    def test_has_11_capabilities(self) -> None:
        assert len(self._pkg.capabilities) == 11

    def test_all_expected_capabilities_present(self) -> None:
        assert set(self._pkg.capabilities) == _GITHUB_CAPS

    def test_has_4_tools(self) -> None:
        assert len(self._pkg.tools) == 4

    def test_tool_names_match_expected(self) -> None:
        assert {t.name for t in self._pkg.tools} == _GITHUB_TOOLS

    def test_github_get_pr_adapter_type(self) -> None:
        pr_tool = next(t for t in self._pkg.tools if t.name == "github_get_pr")
        assert pr_tool.adapter_type == "github"

    def test_github_get_pr_operation(self) -> None:
        pr_tool = next(t for t in self._pkg.tools if t.name == "github_get_pr")
        assert pr_tool.adapter_config["operation"] == "get_pull_request"

    def test_mcp_get_pr_comments_adapter_type(self) -> None:
        mcp_tool = next(t for t in self._pkg.tools if t.name == "mcp_get_pr_comments")
        assert mcp_tool.adapter_type == "mcp"

    def test_has_4_agents(self) -> None:
        assert len(self._pkg.agents) == 4

    def test_agent_ids(self) -> None:
        agent_ids = {a.id for a in self._pkg.agents}
        assert agent_ids == {"pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"}

    def test_has_one_permission(self) -> None:
        assert len(self._pkg.permissions) == 1

    def test_permission_id(self) -> None:
        assert self._pkg.permissions[0].id == "read_github_prs_readonly"

    def test_permission_risk_level(self) -> None:
        assert self._pkg.permissions[0].risk_level == "low"

    def test_dependencies_empty(self) -> None:
        assert self._pkg.dependencies == []

    def test_find_by_capability_fetch_pr_data(self) -> None:
        results = _CATALOG.find_by_capability("fetch_pr_data")
        assert any(p.id == "github-integration" for p in results)

    def test_find_by_capability_synthesize_findings(self) -> None:
        results = _CATALOG.find_by_capability("synthesize_findings")
        assert any(p.id == "github-integration" for p in results)


# ---------------------------------------------------------------------------
# knowledge-search manifest
# ---------------------------------------------------------------------------


class TestKnowledgeSearchManifest:
    @pytest.fixture(autouse=True)
    def pkg(self):
        self._pkg = _CATALOG.get("knowledge-search")

    def test_id(self) -> None:
        assert self._pkg.id == "knowledge-search"

    def test_name(self) -> None:
        assert self._pkg.name == "Knowledge Search"

    def test_version(self) -> None:
        assert self._pkg.version == "1.0.0"

    def test_category(self) -> None:
        assert self._pkg.category == "knowledge"

    def test_provides_static_agent(self) -> None:
        assert "static_agent" in self._pkg.provides

    def test_does_not_provide_runtime_agent(self) -> None:
        assert "runtime_agent" not in self._pkg.provides

    def test_has_3_capabilities(self) -> None:
        assert len(self._pkg.capabilities) == 3

    def test_all_expected_capabilities_present(self) -> None:
        assert set(self._pkg.capabilities) == _KNOWLEDGE_CAPS

    def test_has_1_tool(self) -> None:
        assert len(self._pkg.tools) == 1

    def test_tool_name_is_knowledge_search(self) -> None:
        assert self._pkg.tools[0].name == "knowledge_search"

    def test_knowledge_search_adapter_type(self) -> None:
        assert self._pkg.tools[0].adapter_type == "knowledge"

    def test_knowledge_search_collections(self) -> None:
        cfg = self._pkg.tools[0].adapter_config
        assert "collections" in cfg
        assert "coding-standards" in cfg["collections"]

    def test_agents_empty(self) -> None:
        assert self._pkg.agents == []

    def test_has_one_permission(self) -> None:
        assert len(self._pkg.permissions) == 1

    def test_permission_id(self) -> None:
        assert self._pkg.permissions[0].id == "read_knowledge_base_readonly"

    def test_permission_risk_level(self) -> None:
        assert self._pkg.permissions[0].risk_level == "low"

    def test_dependencies_empty(self) -> None:
        assert self._pkg.dependencies == []

    def test_find_by_capability_search_knowledge(self) -> None:
        results = _CATALOG.find_by_capability("search_knowledge")
        assert any(p.id == "knowledge-search" for p in results)
