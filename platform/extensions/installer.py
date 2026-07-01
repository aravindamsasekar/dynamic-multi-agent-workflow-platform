"""PackageInstaller — registers extension tools/capabilities and persists install state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from sqlalchemy.orm import Session

from platform.core.models.tool import AdapterType, ToolDefinition
from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.models import CapabilityPackage
from platform.persistence.repositories.package_repo import InstalledExtensionStore
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import OperationType, ToolCapabilityDescriptor
from platform.registries.tool_registry import ToolRegistry
from platform.tools.filesystem_adapter import FilesystemAdapter


# ---------------------------------------------------------------------------
# Custom exceptions — translated to HTTP errors by the router
# ---------------------------------------------------------------------------


class ExtensionNotFoundError(Exception):
    def __init__(self, extension_id: str) -> None:
        self.extension_id = extension_id
        super().__init__(f"Extension not found: {extension_id!r}")


class ExtensionAlreadyInstalledError(Exception):
    def __init__(self, extension_id: str) -> None:
        self.extension_id = extension_id
        super().__init__(f"Extension already installed: {extension_id!r}")


class MissingPermissionError(Exception):
    def __init__(self, extension_id: str, missing: list[str]) -> None:
        self.extension_id = extension_id
        self.missing = missing
        super().__init__(
            f"Missing required permissions for {extension_id!r}: {missing}"
        )


class DependencyNotInstalledError(Exception):
    def __init__(self, extension_id: str, dependency_id: str) -> None:
        self.extension_id = extension_id
        self.dependency_id = dependency_id
        super().__init__(
            f"Dependency {dependency_id!r} required by {extension_id!r} is not installed"
        )


class UnknownAdapterTypeError(Exception):
    pass


# ---------------------------------------------------------------------------
# Install result
# ---------------------------------------------------------------------------


@dataclass
class InstallResult:
    extension_id: str
    name: str
    version: str
    capabilities_added: list[str]
    tools_added: list[str]


# ---------------------------------------------------------------------------
# PackageInstaller
# ---------------------------------------------------------------------------


class PackageInstaller:
    """Validates, registers, and persists extension installations.

    Safety boundary: only adapter types listed in ADAPTER_FACTORIES can be
    instantiated. No dynamic imports, no arbitrary code execution.
    """

    # Closed set of supported adapter types for Phase 2.
    # Extended in later phases as new extension categories are added.
    ADAPTER_FACTORIES: ClassVar[dict[str, type]] = {
        "filesystem": FilesystemAdapter,
    }

    def __init__(
        self,
        catalog: ExtensionCatalog,
        store: InstalledExtensionStore,
        tool_registry: ToolRegistry,
        capability_registry: CapabilityRegistry,
    ) -> None:
        self._catalog = catalog
        self._store = store
        self._tool_registry = tool_registry
        self._capability_registry = capability_registry

    def install(
        self,
        extension_id: str,
        permissions_granted: list[str],
        session: Session,
    ) -> InstallResult:
        """Install an extension: validate, register, persist.

        Raises:
            ExtensionNotFoundError: extension_id not in catalog.
            ExtensionAlreadyInstalledError: already in installed_packages.
            MissingPermissionError: caller did not grant all required permissions.
            DependencyNotInstalledError: a listed dependency is not installed.
            UnknownAdapterTypeError: manifest references an unsupported adapter type.
        """
        pkg = self._catalog.get(extension_id)
        if pkg is None:
            raise ExtensionNotFoundError(extension_id)

        if self._store.is_installed(session, extension_id):
            raise ExtensionAlreadyInstalledError(extension_id)

        required_ids = {p.id for p in pkg.permissions}
        granted_ids = set(permissions_granted)
        missing = sorted(required_ids - granted_ids)
        if missing:
            raise MissingPermissionError(extension_id, missing)

        for dep_id in pkg.dependencies:
            if not self._store.is_installed(session, dep_id):
                raise DependencyNotInstalledError(extension_id, dep_id)

        result = self._register_package(pkg)

        self._store.insert(
            session,
            package_id=extension_id,
            version=pkg.version,
            permissions_granted=permissions_granted,
        )
        self._store.record_history(
            session,
            package_id=extension_id,
            action="install",
            permissions=permissions_granted,
        )

        return result

    def restore_from_db(self, session: Session) -> None:
        """Re-register all active packages from DB without writing new rows.

        Called once at startup after V3 registrations are complete.
        Idempotent: safe to call even when tools are already registered
        (ToolRegistry.register() skips duplicates; CapabilityRegistry checks
        are performed before registering).
        """
        for row in self._store.list_active(session):
            pkg = self._catalog.get(row.id)
            if pkg is not None:
                self._register_package(pkg)

    def _register_package(self, pkg: CapabilityPackage) -> InstallResult:
        """Register tools and capabilities for a package. Idempotent.

        Shared by install() and restore_from_db(). Does not write to the DB.
        """
        tools_added: list[str] = []

        for tool in pkg.tools:
            adapter_cls = self.ADAPTER_FACTORIES.get(tool.adapter_type)
            if adapter_cls is None:
                raise UnknownAdapterTypeError(
                    f"No adapter factory for adapter_type {tool.adapter_type!r} "
                    f"(package {pkg.id!r}). Supported types: {sorted(self.ADAPTER_FACTORIES)}"
                )

            # ToolRegistry.register() is idempotent — skips if already registered.
            self._tool_registry.register(
                tool.name,
                adapter_cls(),
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    adapter_type=AdapterType(tool.adapter_type),
                    adapter_config=tool.adapter_config,
                ),
            )
            tools_added.append(tool.name)

            # CapabilityRegistry.register_tool() raises on duplicates, so check first.
            if self._capability_registry.get_tool(tool.name) is None:
                self._capability_registry.register_tool(ToolCapabilityDescriptor(
                    tool_name=tool.name,
                    name=tool.name.replace("_", " ").title(),
                    description=tool.description,
                    capabilities=list(pkg.capabilities),
                    operation_type=OperationType.READ,
                    data_source=tool.adapter_type,
                ))

        # register_generatable_capability() is already idempotent (uses a set).
        if "runtime_agent" in pkg.provides:
            for cap in pkg.capabilities:
                self._capability_registry.register_generatable_capability(cap)

        return InstallResult(
            extension_id=pkg.id,
            name=pkg.name,
            version=pkg.version,
            capabilities_added=list(pkg.capabilities),
            tools_added=tools_added,
        )
