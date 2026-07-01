"""Phase 5 API tests — tool_names in catalog, auto_installed flag, default extensions.

Includes the marketplace metadata consistency regression test:
  Manifest → declares tools → Marketplace API → reports those tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import (
    get_db_session,
    get_extension_catalog,
    get_installed_extension_store,
    get_package_installer,
)
from api.routers import extensions as extensions_router
from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.installer import PackageInstaller
from platform.persistence.database import Base
from platform.persistence.repositories.package_repo import InstalledExtensionStore
from platform.planner.capability_registry import CapabilityRegistry
from platform.registries.tool_registry import ToolRegistry

_EXTENSIONS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "resources" / "extensions"
)

_CATALOG = ExtensionCatalog.load(_EXTENSIONS_DIR)

_FS_INSTALL_BODY = {
    "extension_id": "filesystem-reader",
    "permissions_granted": ["read_local_files_readonly"],
}

_GITHUB_INSTALL_BODY = {
    "extension_id": "github-integration",
    "permissions_granted": ["read_github_prs_readonly"],
}

_KNOWLEDGE_INSTALL_BODY = {
    "extension_id": "knowledge-search",
    "permissions_granted": ["read_knowledge_base_readonly"],
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def test_app(db_factory):
    """FastAPI test app with real 3-extension catalog, in-memory SQLite, fresh registries."""
    store = InstalledExtensionStore()
    tool_reg = ToolRegistry()
    cap_reg = CapabilityRegistry()
    installer = PackageInstaller(_CATALOG, store, tool_reg, cap_reg)

    def _session():
        with db_factory() as s:
            yield s

    app = FastAPI()
    app.include_router(extensions_router.router, prefix="/extensions", tags=["extensions"])
    app.dependency_overrides[get_extension_catalog] = lambda: _CATALOG
    app.dependency_overrides[get_installed_extension_store] = lambda: store
    app.dependency_overrides[get_db_session] = _session
    app.dependency_overrides[get_package_installer] = lambda: installer
    return app


# ---------------------------------------------------------------------------
# TestCatalogNowHasThreeExtensions
# ---------------------------------------------------------------------------


class TestCatalogNowHasThreeExtensions:
    async def test_list_returns_three_extensions(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        data = response.json()
        assert len(data["extensions"]) == 3

    async def test_github_integration_in_catalog(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ids = [e["id"] for e in response.json()["extensions"]]
        assert "github-integration" in ids

    async def test_knowledge_search_in_catalog(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ids = [e["id"] for e in response.json()["extensions"]]
        assert "knowledge-search" in ids

    async def test_filesystem_reader_still_in_catalog(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ids = [e["id"] for e in response.json()["extensions"]]
        assert "filesystem-reader" in ids


# ---------------------------------------------------------------------------
# TestToolNamesInCatalogResponse
# ---------------------------------------------------------------------------


class TestToolNamesInCatalogResponse:
    """Verify tool_names field is populated correctly in GET /extensions."""

    async def test_filesystem_reader_reports_tool_names(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ext = next(e for e in response.json()["extensions"] if e["id"] == "filesystem-reader")
        assert ext["tool_names"] == ["filesystem_read_file"]

    async def test_github_integration_reports_tool_names(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ext = next(e for e in response.json()["extensions"] if e["id"] == "github-integration")
        assert set(ext["tool_names"]) == {
            "github_get_pr", "github_get_files", "github_get_diff", "mcp_get_pr_comments"
        }

    async def test_knowledge_search_reports_tool_names(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ext = next(e for e in response.json()["extensions"] if e["id"] == "knowledge-search")
        assert ext["tool_names"] == ["knowledge_search"]

    async def test_tool_names_field_present_for_all_extensions(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        for ext in response.json()["extensions"]:
            assert "tool_names" in ext, f"tool_names missing for {ext['id']}"


# ---------------------------------------------------------------------------
# TestDefaultExtensionsInstalledFalseWithoutAutoInstall
# (API tests don't call initialize(); auto-install is not triggered here.)
# ---------------------------------------------------------------------------


class TestDefaultExtensionsInstalledFalseWithoutAutoInstall:
    async def test_github_integration_installed_false_initially(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ext = next(e for e in response.json()["extensions"] if e["id"] == "github-integration")
        assert ext["installed"] is False

    async def test_knowledge_search_installed_false_initially(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        ext = next(e for e in response.json()["extensions"] if e["id"] == "knowledge-search")
        assert ext["installed"] is False

    async def test_installed_list_empty_without_auto_install(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions/installed")
        assert response.json() == {"extensions": []}


# ---------------------------------------------------------------------------
# TestAutoInstalledFieldInInstalledResponse
# ---------------------------------------------------------------------------


class TestAutoInstalledFieldInInstalledResponse:
    async def test_auto_installed_false_for_user_install(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_FS_INSTALL_BODY)
            response = await client.get("/extensions/installed")
        ext = response.json()["extensions"][0]
        assert ext["auto_installed"] is False

    async def test_auto_installed_true_after_simulated_auto_install(
        self, test_app, db_factory
    ) -> None:
        # Simulate auto-install by directly calling install with auto_installed=True.
        store = InstalledExtensionStore()
        tool_reg = ToolRegistry()
        cap_reg = CapabilityRegistry()
        installer = PackageInstaller(_CATALOG, store, tool_reg, cap_reg)

        with db_factory() as session:
            installer.install(
                "github-integration",
                ["read_github_prs_readonly"],
                session,
                auto_installed=True,
            )
            session.commit()

        def _session():
            with db_factory() as s:
                yield s

        app = FastAPI()
        app.include_router(extensions_router.router, prefix="/extensions", tags=["extensions"])
        app.dependency_overrides[get_extension_catalog] = lambda: _CATALOG
        app.dependency_overrides[get_installed_extension_store] = lambda: store
        app.dependency_overrides[get_db_session] = _session
        app.dependency_overrides[get_package_installer] = lambda: installer

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/extensions/installed")

        ext = next(
            e for e in response.json()["extensions"] if e["id"] == "github-integration"
        )
        assert ext["auto_installed"] is True

    async def test_auto_installed_field_present_in_response(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_FS_INSTALL_BODY)
            response = await client.get("/extensions/installed")
        ext = response.json()["extensions"][0]
        assert "auto_installed" in ext

    async def test_github_installed_true_after_install(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_GITHUB_INSTALL_BODY)
            response = await client.get("/extensions")
        ext = next(e for e in response.json()["extensions"] if e["id"] == "github-integration")
        assert ext["installed"] is True

    async def test_github_duplicate_install_returns_409(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_GITHUB_INSTALL_BODY)
            response = await client.post("/extensions/install", json=_GITHUB_INSTALL_BODY)
        assert response.status_code == 409

    async def test_knowledge_install_succeeds(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post("/extensions/install", json=_KNOWLEDGE_INSTALL_BODY)
        assert response.status_code == 201
        data = response.json()
        assert data["extension_id"] == "knowledge-search"
        assert set(data["capabilities_added"]) == {
            "search_knowledge", "search_coding_standards", "search_architecture"
        }

    async def test_github_install_succeeds_with_capabilities(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post("/extensions/install", json=_GITHUB_INSTALL_BODY)
        assert response.status_code == 201
        data = response.json()
        assert data["extension_id"] == "github-integration"
        assert "fetch_pr_data" in data["capabilities_added"]
        assert "synthesize_findings" in data["capabilities_added"]


# ---------------------------------------------------------------------------
# TestMarketplaceMetadataConsistency
# (Additional Regression Test: Manifest → declares tools → API reports those tools)
# ---------------------------------------------------------------------------


class TestMarketplaceMetadataConsistency:
    """Verify that tool_names in the API response exactly matches what is declared in
    the extension manifest. This prevents manifest drift over time.
    """

    async def _get_extension(self, test_app, ext_id: str) -> dict:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        return next(e for e in response.json()["extensions"] if e["id"] == ext_id)

    async def test_filesystem_reader_tools_match_manifest(self, test_app) -> None:
        manifest_tools = {t.name for t in _CATALOG.get("filesystem-reader").tools}
        api_ext = await self._get_extension(test_app, "filesystem-reader")
        assert set(api_ext["tool_names"]) == manifest_tools

    async def test_github_integration_tools_match_manifest(self, test_app) -> None:
        manifest_tools = {t.name for t in _CATALOG.get("github-integration").tools}
        api_ext = await self._get_extension(test_app, "github-integration")
        assert set(api_ext["tool_names"]) == manifest_tools

    async def test_knowledge_search_tools_match_manifest(self, test_app) -> None:
        manifest_tools = {t.name for t in _CATALOG.get("knowledge-search").tools}
        api_ext = await self._get_extension(test_app, "knowledge-search")
        assert set(api_ext["tool_names"]) == manifest_tools

    async def test_github_integration_capabilities_match_manifest(self, test_app) -> None:
        manifest_caps = set(_CATALOG.get("github-integration").capabilities)
        api_ext = await self._get_extension(test_app, "github-integration")
        assert set(api_ext["capabilities"]) == manifest_caps

    async def test_knowledge_search_capabilities_match_manifest(self, test_app) -> None:
        manifest_caps = set(_CATALOG.get("knowledge-search").capabilities)
        api_ext = await self._get_extension(test_app, "knowledge-search")
        assert set(api_ext["capabilities"]) == manifest_caps

    async def test_all_extensions_tool_names_match_manifests(self, test_app) -> None:
        """Cross-check all extensions at once — catches any new extension that drifts."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        for api_ext in response.json()["extensions"]:
            ext_id = api_ext["id"]
            pkg = _CATALOG.get(ext_id)
            assert pkg is not None, f"API returned extension {ext_id!r} not in catalog"
            expected_tools = {t.name for t in pkg.tools}
            actual_tools = set(api_ext["tool_names"])
            assert actual_tools == expected_tools, (
                f"Tool names mismatch for {ext_id!r}: "
                f"manifest={sorted(expected_tools)}, api={sorted(actual_tools)}"
            )
