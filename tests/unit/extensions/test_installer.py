"""Unit tests for PackageInstaller — install, validation, and restore behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.installer import (
    DependencyNotInstalledError,
    ExtensionAlreadyInstalledError,
    ExtensionNotFoundError,
    MissingPermissionError,
    PackageInstaller,
    UnknownAdapterTypeError,
)
from platform.persistence.database import Base
from platform.persistence.models import InstallHistoryRow, InstalledPackageRow
from platform.persistence.repositories.package_repo import InstalledExtensionStore
from platform.planner.capability_registry import CapabilityRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.tools.filesystem_adapter import FilesystemAdapter

# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

_FS_MANIFEST = """\
id: filesystem-reader
name: Filesystem Reader
version: "1.0.0"
description: Read local files.
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
        path:
          type: string
      required: [path]
agents: []
agent_prompts: []
permissions:
  - id: read_local_files_readonly
    display_name: Read local files (read-only)
    description: Allows reading local files.
    risk_level: low
"""

_DEP_MANIFEST = """\
id: dependent-pkg
name: Dependent Package
version: "1.0.0"
description: Depends on filesystem-reader.
category: developer
provides:
  - runtime_agent
dependencies:
  - filesystem-reader
capabilities:
  - dependent_cap
tools: []
agents: []
agent_prompts: []
permissions:
  - id: dep_perm
    display_name: Dep Permission
    description: Some permission.
    risk_level: low
"""

_UNKNOWN_ADAPTER_MANIFEST = """\
id: bad-adapter-pkg
name: Bad Adapter Package
version: "1.0.0"
description: Uses unsupported adapter.
category: developer
provides:
  - runtime_agent
dependencies: []
capabilities:
  - some_cap
tools:
  - name: some_tool
    description: A tool with unknown adapter.
    adapter_type: unknown_adapter
    adapter_config: {}
    input_schema:
      type: object
      properties: {}
agents: []
agent_prompts: []
permissions:
  - id: some_perm
    display_name: Some Permission
    description: A permission.
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
def fs_catalog(tmp_path: Path) -> ExtensionCatalog:
    _write(tmp_path, "filesystem-reader.yaml", _FS_MANIFEST)
    return ExtensionCatalog.load(tmp_path)


@pytest.fixture
def catalog_with_dep(tmp_path: Path) -> ExtensionCatalog:
    _write(tmp_path, "filesystem-reader.yaml", _FS_MANIFEST)
    _write(tmp_path, "dependent-pkg.yaml", _DEP_MANIFEST)
    return ExtensionCatalog.load(tmp_path)


@pytest.fixture
def bad_adapter_catalog(tmp_path: Path) -> ExtensionCatalog:
    _write(tmp_path, "bad-adapter-pkg.yaml", _UNKNOWN_ADAPTER_MANIFEST)
    return ExtensionCatalog.load(tmp_path)


def _make_installer(catalog: ExtensionCatalog) -> tuple[PackageInstaller, ToolRegistry, CapabilityRegistry]:
    store = InstalledExtensionStore()
    tool_reg = ToolRegistry()
    cap_reg = CapabilityRegistry()
    installer = PackageInstaller(catalog, store, tool_reg, cap_reg)
    return installer, tool_reg, cap_reg


# ---------------------------------------------------------------------------
# TestPackageInstallerInstall
# ---------------------------------------------------------------------------


class TestPackageInstallerInstall:
    def test_install_filesystem_reader_succeeds(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            result = installer.install(
                "filesystem-reader",
                ["read_local_files_readonly"],
                session,
            )
            session.commit()
        assert result.extension_id == "filesystem-reader"

    def test_install_returns_correct_result(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            result = installer.install(
                "filesystem-reader",
                ["read_local_files_readonly"],
                session,
            )
            session.commit()
        assert result.name == "Filesystem Reader"
        assert result.version == "1.0.0"
        assert result.capabilities_added == ["filesystem_read"]
        assert result.tools_added == ["filesystem_read_file"]

    def test_install_registers_tool_in_tool_registry(self, fs_catalog, db_factory) -> None:
        installer, tool_reg, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        assert tool_reg.exists("filesystem_read_file")

    def test_install_tool_adapter_is_filesystem(self, fs_catalog, db_factory) -> None:
        installer, tool_reg, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        assert isinstance(tool_reg.get("filesystem_read_file"), FilesystemAdapter)

    def test_install_registers_tool_descriptor_in_capability_registry(self, fs_catalog, db_factory) -> None:
        installer, _, cap_reg = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        assert cap_reg.get_tool("filesystem_read_file") is not None

    def test_install_registers_generatable_capability(self, fs_catalog, db_factory) -> None:
        installer, _, cap_reg = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        assert "filesystem_read" in cap_reg.all_capabilities()

    def test_install_persists_package_row(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        with db_factory() as session:
            rows = session.scalars(select(InstalledPackageRow)).all()
        assert len(rows) == 1
        assert rows[0].id == "filesystem-reader"
        assert rows[0].status == "active"

    def test_install_persists_history_row(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        with db_factory() as session:
            rows = session.scalars(select(InstallHistoryRow)).all()
        assert len(rows) == 1
        assert rows[0].package_id == "filesystem-reader"
        assert rows[0].action == "install"

    def test_duplicate_install_raises(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()
        with pytest.raises(ExtensionAlreadyInstalledError):
            with db_factory() as session:
                installer.install("filesystem-reader", ["read_local_files_readonly"], session)

    def test_unknown_extension_raises(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with pytest.raises(ExtensionNotFoundError):
            with db_factory() as session:
                installer.install("nonexistent-pkg", [], session)

    def test_missing_permission_raises(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with pytest.raises(MissingPermissionError) as exc_info:
            with db_factory() as session:
                installer.install("filesystem-reader", [], session)
        assert "read_local_files_readonly" in exc_info.value.missing

    def test_dependency_missing_raises(self, catalog_with_dep, db_factory) -> None:
        installer, _, _ = _make_installer(catalog_with_dep)
        with pytest.raises(DependencyNotInstalledError) as exc_info:
            with db_factory() as session:
                installer.install("dependent-pkg", ["dep_perm"], session)
        assert exc_info.value.dependency_id == "filesystem-reader"

    def test_unknown_adapter_type_raises(self, bad_adapter_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(bad_adapter_catalog)
        with pytest.raises(UnknownAdapterTypeError):
            with db_factory() as session:
                installer.install("bad-adapter-pkg", ["some_perm"], session)


# ---------------------------------------------------------------------------
# TestPackageInstallerRestoreFromDb
# ---------------------------------------------------------------------------


class TestPackageInstallerRestoreFromDb:
    def test_restore_registers_tool_in_tool_registry(self, fs_catalog, db_factory) -> None:
        installer, tool_reg, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()

        # New registries simulate a server restart
        store = InstalledExtensionStore()
        fresh_tool_reg = ToolRegistry()
        fresh_cap_reg = CapabilityRegistry()
        fresh_installer = PackageInstaller(fs_catalog, store, fresh_tool_reg, fresh_cap_reg)

        with db_factory() as session:
            fresh_installer.restore_from_db(session)

        assert fresh_tool_reg.exists("filesystem_read_file")

    def test_restore_registers_generatable_capability(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()

        store = InstalledExtensionStore()
        fresh_tool_reg = ToolRegistry()
        fresh_cap_reg = CapabilityRegistry()
        fresh_installer = PackageInstaller(fs_catalog, store, fresh_tool_reg, fresh_cap_reg)

        with db_factory() as session:
            fresh_installer.restore_from_db(session)

        assert "filesystem_read" in fresh_cap_reg.all_capabilities()

    def test_restore_does_not_write_new_db_rows(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()

        with db_factory() as session:
            count_before = len(session.scalars(select(InstalledPackageRow)).all())

        # restore_from_db must not write new rows
        with db_factory() as session:
            installer.restore_from_db(session)

        with db_factory() as session:
            count_after = len(session.scalars(select(InstalledPackageRow)).all())

        assert count_before == count_after == 1

    def test_restore_is_idempotent(self, fs_catalog, db_factory) -> None:
        installer, _, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.install("filesystem-reader", ["read_local_files_readonly"], session)
            session.commit()

        # Calling restore twice must not raise
        with db_factory() as session:
            installer.restore_from_db(session)
        with db_factory() as session:
            installer.restore_from_db(session)

    def test_restore_on_empty_db_does_nothing(self, fs_catalog, db_factory) -> None:
        installer, tool_reg, _ = _make_installer(fs_catalog)
        with db_factory() as session:
            installer.restore_from_db(session)
        assert not tool_reg.exists("filesystem_read_file")


# ---------------------------------------------------------------------------
# TestAdapterFactory
# ---------------------------------------------------------------------------


class TestAdapterFactory:
    def test_adapter_factory_contains_filesystem(self) -> None:
        assert "filesystem" in PackageInstaller.ADAPTER_FACTORIES

    def test_adapter_factory_is_finite_set(self) -> None:
        # All values must be importable classes, not dynamic lookups
        for key, cls in PackageInstaller.ADAPTER_FACTORIES.items():
            assert isinstance(key, str)
            assert callable(cls)

    def test_filesystem_factory_produces_filesystem_adapter(self) -> None:
        cls = PackageInstaller.ADAPTER_FACTORIES["filesystem"]
        adapter = cls()
        assert isinstance(adapter, FilesystemAdapter)
