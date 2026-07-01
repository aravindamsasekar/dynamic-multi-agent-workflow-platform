"""Unit tests for the /extensions API endpoints.

Uses a test-only FastAPI app (no lifespan) with dependency overrides and
in-memory SQLite so no real filesystem or LLM is needed.
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

_INSTALL_BODY = {
    "extension_id": "filesystem-reader",
    "permissions_granted": ["read_local_files_readonly"],
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
    """FastAPI test app with real catalog + in-memory SQLite and fresh registries."""
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
# GET /extensions — catalog listing
# ---------------------------------------------------------------------------


class TestListExtensions:
    async def test_returns_200(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        assert response.status_code == 200

    async def test_includes_filesystem_reader(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        data = response.json()
        ids = [e["id"] for e in data["extensions"]]
        assert "filesystem-reader" in ids

    async def test_filesystem_reader_installed_false_initially(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        data = response.json()
        fs = next(e for e in data["extensions"] if e["id"] == "filesystem-reader")
        assert fs["installed"] is False

    async def test_filesystem_reader_installed_true_after_install(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_INSTALL_BODY)
            response = await client.get("/extensions")
        data = response.json()
        fs = next(e for e in data["extensions"] if e["id"] == "filesystem-reader")
        assert fs["installed"] is True

    async def test_permission_fields_present(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions")
        data = response.json()
        fs = next(e for e in data["extensions"] if e["id"] == "filesystem-reader")
        assert len(fs["permissions"]) >= 1
        perm = fs["permissions"][0]
        assert "id" in perm
        assert "display_name" in perm
        assert "description" in perm
        assert perm["risk_level"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# GET /extensions/installed
# ---------------------------------------------------------------------------


class TestListInstalledExtensions:
    async def test_returns_200(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions/installed")
        assert response.status_code == 200

    async def test_empty_initially(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/extensions/installed")
        assert response.json() == {"extensions": []}

    async def test_includes_filesystem_reader_after_install(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_INSTALL_BODY)
            response = await client.get("/extensions/installed")
        data = response.json()
        ids = [e["id"] for e in data["extensions"]]
        assert "filesystem-reader" in ids

    async def test_installed_response_fields(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_INSTALL_BODY)
            response = await client.get("/extensions/installed")
        ext = response.json()["extensions"][0]
        assert ext["id"] == "filesystem-reader"
        assert ext["name"] == "Filesystem Reader"
        assert ext["version"] == "1.0.0"
        assert "installed_at" in ext
        assert "filesystem_read" in ext["capabilities_active"]
        assert "read_local_files_readonly" in ext["permissions_granted"]


# ---------------------------------------------------------------------------
# POST /extensions/install
# ---------------------------------------------------------------------------


class TestInstallExtension:
    async def test_install_returns_201(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post("/extensions/install", json=_INSTALL_BODY)
        assert response.status_code == 201

    async def test_install_response_fields(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post("/extensions/install", json=_INSTALL_BODY)
        data = response.json()
        assert data["extension_id"] == "filesystem-reader"
        assert data["name"] == "Filesystem Reader"
        assert data["version"] == "1.0.0"
        assert "filesystem_read" in data["capabilities_added"]
        assert "filesystem_read_file" in data["tools_added"]

    async def test_duplicate_install_returns_409(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/extensions/install", json=_INSTALL_BODY)
            response = await client.post("/extensions/install", json=_INSTALL_BODY)
        assert response.status_code == 409

    async def test_unknown_extension_returns_404(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/extensions/install",
                json={"extension_id": "nonexistent", "permissions_granted": []},
            )
        assert response.status_code == 404

    async def test_missing_permission_returns_422(self, test_app) -> None:
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/extensions/install",
                json={"extension_id": "filesystem-reader", "permissions_granted": []},
            )
        assert response.status_code == 422
