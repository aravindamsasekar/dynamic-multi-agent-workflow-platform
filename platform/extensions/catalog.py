"""ExtensionCatalog — loads and queries capability extension manifests from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from platform.extensions.models import (
    AgentManifest,
    AgentPromptTemplate,
    CapabilityPackage,
    PackagePermission,
    ToolManifest,
)

_VALID_RISK_LEVELS = frozenset({"low", "medium", "high"})


class ExtensionCatalogError(Exception):
    """Raised when a manifest is missing required fields or has invalid content."""


class ExtensionCatalog:
    """Immutable catalog of capability extension manifests loaded from YAML files.

    Constructed once at startup via ExtensionCatalog.load(). The catalog is never
    mutated at runtime — installation state lives in InstalledExtensionStore (Phase 2).
    """

    def __init__(self, packages: dict[str, CapabilityPackage]) -> None:
        self._by_id = packages
        # Inverted index: capability string → packages that provide it
        self._by_capability: dict[str, list[CapabilityPackage]] = {}
        for pkg in packages.values():
            for cap in pkg.capabilities:
                self._by_capability.setdefault(cap, []).append(pkg)

    @classmethod
    def load(cls, directory: Path) -> ExtensionCatalog:
        """Load all *.yaml files from directory. Raises ExtensionCatalogError on invalid manifests."""
        packages: dict[str, CapabilityPackage] = {}
        for yaml_file in sorted(directory.glob("*.yaml")):
            pkg = _parse_manifest(yaml_file)
            if pkg.id in packages:
                raise ExtensionCatalogError(
                    f"Duplicate package id {pkg.id!r} found in {yaml_file}"
                )
            packages[pkg.id] = pkg
        return cls(packages)

    def all(self) -> list[CapabilityPackage]:
        return list(self._by_id.values())

    def get(self, package_id: str) -> CapabilityPackage | None:
        return self._by_id.get(package_id)

    def find_by_capability(self, capability: str) -> list[CapabilityPackage]:
        return list(self._by_capability.get(capability, []))


def _parse_manifest(path: Path) -> CapabilityPackage:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExtensionCatalogError(f"Failed to parse {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ExtensionCatalogError(f"Manifest {path} must be a YAML mapping")

    _require_fields(data, path, [
        "id", "name", "version", "description", "category",
        "provides", "dependencies", "capabilities", "tools", "permissions",
    ])

    return CapabilityPackage(
        id=data["id"],
        name=data["name"],
        version=str(data["version"]),
        description=data["description"],
        category=data["category"],
        provides=list(data["provides"]),
        dependencies=list(data["dependencies"]),
        capabilities=list(data["capabilities"]),
        tools=[_parse_tool(t, path) for t in data.get("tools", [])],
        agents=[_parse_agent(a, path) for a in data.get("agents", [])],
        agent_prompts=[
            AgentPromptTemplate(
                capability=ap["capability"],
                system_prompt=ap["system_prompt"],
            )
            for ap in data.get("agent_prompts", [])
        ],
        permissions=[_parse_permission(p, path) for p in data["permissions"]],
    )


def _require_fields(data: dict, path: Path, fields: list[str]) -> None:
    missing = [f for f in fields if f not in data]
    if missing:
        raise ExtensionCatalogError(
            f"Manifest {path} missing required fields: {missing}"
        )


def _parse_tool(data: dict, path: Path) -> ToolManifest:
    _require_fields(data, path, ["name", "description", "adapter_type", "adapter_config", "input_schema"])
    return ToolManifest(
        name=data["name"],
        description=data["description"],
        adapter_type=data["adapter_type"],
        adapter_config=dict(data["adapter_config"] or {}),
        input_schema=dict(data["input_schema"]),
    )


def _parse_agent(data: dict, path: Path) -> AgentManifest:
    _require_fields(data, path, ["id", "name", "description", "capabilities"])
    return AgentManifest(
        id=data["id"],
        name=data["name"],
        description=data["description"],
        capabilities=list(data["capabilities"]),
        consumes=list(data.get("consumes", [])),
        produces=list(data.get("produces", [])),
    )


def _parse_permission(data: dict, path: Path) -> PackagePermission:
    _require_fields(data, path, ["id", "display_name", "description", "risk_level"])
    risk = data["risk_level"]
    if risk not in _VALID_RISK_LEVELS:
        raise ExtensionCatalogError(
            f"Manifest {path}: permission {data['id']!r} has invalid risk_level {risk!r} "
            f"(must be one of: {sorted(_VALID_RISK_LEVELS)})"
        )
    return PackagePermission(
        id=data["id"],
        display_name=data["display_name"],
        description=data["description"],
        risk_level=risk,
    )
