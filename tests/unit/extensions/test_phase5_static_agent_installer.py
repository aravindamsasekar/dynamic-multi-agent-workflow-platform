"""Phase 5 installer tests — static_agent skip and auto_installed flag."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.installer import PackageInstaller
from platform.persistence.database import Base
from platform.persistence.models import InstalledPackageRow
from platform.persistence.repositories.package_repo import InstalledExtensionStore
from platform.planner.capability_registry import CapabilityRegistry
from platform.registries.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# Minimal manifests used in tests
# ---------------------------------------------------------------------------

_STATIC_AGENT_MANIFEST = """\
id: github-integration
name: GitHub Integration
version: "1.0.0"
description: Static agent extension for PR review.
category: github
provides:
  - static_agent
dependencies: []
capabilities:
  - fetch_pr_data
  - review_code_quality
tools:
  - name: github_get_pr
    description: Fetches PR metadata.
    adapter_type: github
    adapter_config:
      operation: get_pull_request
    input_schema:
      type: object
      properties:
        owner: {type: string}
        repo: {type: string}
        pull_number: {type: integer}
      required: [owner, repo, pull_number]
agents:
  - id: pr_data_agent
    name: PR Data Agent
    description: Fetches PR data.
    capabilities:
      - fetch_pr_data
    consumes: []
    produces:
      - pr_metadata
agent_prompts: []
permissions:
  - id: read_github_prs_readonly
    display_name: Read GitHub PRs (read-only)
    description: Allows reading GitHub pull requests.
    risk_level: low
"""

_RUNTIME_AGENT_MANIFEST = """\
id: filesystem-reader
name: Filesystem Reader
version: "1.0.0"
description: Runtime agent extension.
category: filesystem
provides:
  - runtime_agent
dependencies: []
capabilities:
  - filesystem_read
tools:
  - name: filesystem_read_file
    description: Reads a file.
    adapter_type: filesystem
    adapter_config: {}
    input_schema:
      type: object
      properties:
        path: {type: string}
      required: [path]
agents: []
agent_prompts: []
permissions:
  - id: read_local_files_readonly
    display_name: Read local files (read-only)
    description: Allows reading local files.
    risk_level: low
"""


def _write(tmp_path: Path, filename: str, content: str) -> None:
    (tmp_path / filename).write_text(content, encoding="utf-8")


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
def static_catalog(tmp_path: Path) -> ExtensionCatalog:
    _write(tmp_path, "github-integration.yaml", _STATIC_AGENT_MANIFEST)
    return ExtensionCatalog.load(tmp_path)


@pytest.fixture
def runtime_catalog(tmp_path: Path) -> ExtensionCatalog:
    _write(tmp_path, "filesystem-reader.yaml", _RUNTIME_AGENT_MANIFEST)
    return ExtensionCatalog.load(tmp_path)


@pytest.fixture
def mixed_catalog(tmp_path: Path) -> ExtensionCatalog:
    _write(tmp_path, "github-integration.yaml", _STATIC_AGENT_MANIFEST)
    _write(tmp_path, "filesystem-reader.yaml", _RUNTIME_AGENT_MANIFEST)
    return ExtensionCatalog.load(tmp_path)


def _make_installer(catalog: ExtensionCatalog) -> tuple[PackageInstaller, ToolRegistry, CapabilityRegistry]:
    store = InstalledExtensionStore()
    tool_reg = ToolRegistry()
    cap_reg = CapabilityRegistry()
    installer = PackageInstaller(catalog, store, tool_reg, cap_reg)
    return installer, tool_reg, cap_reg


# ---------------------------------------------------------------------------
# TestStaticAgentSkipsRuntimeRegistration
# ---------------------------------------------------------------------------


class TestStaticAgentSkipsRuntimeRegistration:
    def test_install_static_agent_does_not_register_tool_in_tool_registry(
        self, static_catalog, db_factory
    ) -> None:
        installer, tool_reg, _ = _make_installer(static_catalog)
        with db_factory() as session:
            installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()
        # Tool declared in manifest must NOT be registered — V3 path owns this.
        assert not tool_reg.exists("github_get_pr")

    def test_install_static_agent_does_not_add_capability_to_registry(
        self, static_catalog, db_factory
    ) -> None:
        installer, _, cap_reg = _make_installer(static_catalog)
        with db_factory() as session:
            installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()
        assert "fetch_pr_data" not in cap_reg.all_capabilities()

    def test_install_static_agent_does_not_register_generatable_capability(
        self, static_catalog, db_factory
    ) -> None:
        installer, _, cap_reg = _make_installer(static_catalog)
        with db_factory() as session:
            installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()
        # all_capabilities returns agent caps + generatable caps; neither should appear.
        assert cap_reg.all_capabilities() == []

    def test_install_static_agent_persists_db_row(self, static_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(static_catalog)
        with db_factory() as session:
            installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()
        with db_factory() as session:
            rows = session.scalars(select(InstalledPackageRow)).all()
        assert len(rows) == 1
        assert rows[0].id == "github-integration"
        assert rows[0].status == "active"

    def test_install_static_agent_returns_capabilities_added(
        self, static_catalog, db_factory
    ) -> None:
        installer, _, _ = _make_installer(static_catalog)
        with db_factory() as session:
            result = installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()
        assert set(result.capabilities_added) == {"fetch_pr_data", "review_code_quality"}

    def test_install_static_agent_returns_tool_names_as_metadata(
        self, static_catalog, db_factory
    ) -> None:
        installer, _, _ = _make_installer(static_catalog)
        with db_factory() as session:
            result = installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()
        # tools_added lists declared tools as metadata — not an indicator of registration.
        assert result.tools_added == ["github_get_pr"]

    def test_restore_from_db_static_agent_does_not_register_tools(
        self, static_catalog, db_factory
    ) -> None:
        installer, _, _ = _make_installer(static_catalog)
        with db_factory() as session:
            installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()

        # Simulate restart with fresh registries.
        store = InstalledExtensionStore()
        fresh_tool_reg = ToolRegistry()
        fresh_cap_reg = CapabilityRegistry()
        fresh_installer = PackageInstaller(static_catalog, store, fresh_tool_reg, fresh_cap_reg)

        with db_factory() as session:
            fresh_installer.restore_from_db(session)

        assert not fresh_tool_reg.exists("github_get_pr")
        assert fresh_cap_reg.all_capabilities() == []


# ---------------------------------------------------------------------------
# TestAutoInstalledFlag
# ---------------------------------------------------------------------------


class TestAutoInstalledFlag:
    def test_auto_installed_true_stored_in_db(self, static_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(static_catalog)
        with db_factory() as session:
            installer.install(
                "github-integration",
                ["read_github_prs_readonly"],
                session,
                auto_installed=True,
            )
            session.commit()
        with db_factory() as session:
            row = session.get(InstalledPackageRow, "github-integration")
        assert row is not None
        assert bool(row.auto_installed) is True

    def test_auto_installed_false_by_default(self, static_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(static_catalog)
        with db_factory() as session:
            installer.install("github-integration", ["read_github_prs_readonly"], session)
            session.commit()
        with db_factory() as session:
            row = session.get(InstalledPackageRow, "github-integration")
        assert bool(row.auto_installed) is False

    def test_auto_installed_false_for_runtime_agent(
        self, runtime_catalog, db_factory
    ) -> None:
        installer, _, _ = _make_installer(runtime_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        with db_factory() as session:
            row = session.get(InstalledPackageRow, "filesystem-reader")
        assert bool(row.auto_installed) is False

    def test_auto_installed_true_for_runtime_agent_when_passed(
        self, runtime_catalog, db_factory
    ) -> None:
        installer, _, _ = _make_installer(runtime_catalog)
        with db_factory() as session:
            installer.install(
                "filesystem-reader",
                ["read_local_files_readonly"],
                session,
                auto_installed=True,
            )
            session.commit()
        with db_factory() as session:
            row = session.get(InstalledPackageRow, "filesystem-reader")
        assert bool(row.auto_installed) is True


# ---------------------------------------------------------------------------
# TestRuntimeAgentUnchanged (regression)
# ---------------------------------------------------------------------------


class TestRuntimeAgentUnchanged:
    def test_runtime_agent_still_registers_tool(
        self, runtime_catalog, db_factory
    ) -> None:
        installer, tool_reg, _ = _make_installer(runtime_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        assert tool_reg.exists("filesystem_read_file")

    def test_runtime_agent_still_registers_generatable_capability(
        self, runtime_catalog, db_factory
    ) -> None:
        installer, _, cap_reg = _make_installer(runtime_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        assert "filesystem_read" in cap_reg.all_capabilities()

    def test_static_and_runtime_agent_coexist(
        self, mixed_catalog, db_factory
    ) -> None:
        installer, tool_reg, cap_reg = _make_installer(mixed_catalog)
        with db_factory() as session:
            installer.install("github-integration", ["read_github_prs_readonly"], session)
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        # Static: no tool/cap registration.
        assert not tool_reg.exists("github_get_pr")
        assert "fetch_pr_data" not in cap_reg.all_capabilities()
        # Runtime: tool + generatable cap registered.
        assert tool_reg.exists("filesystem_read_file")
        assert "filesystem_read" in cap_reg.all_capabilities()
