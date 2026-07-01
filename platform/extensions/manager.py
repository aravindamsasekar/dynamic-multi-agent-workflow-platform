"""CapabilityManager — single facade between the planner and the extension marketplace."""

from __future__ import annotations

from dataclasses import dataclass, field

from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.models import InstallSuggestion, PermissionSummary
from platform.planner.capability_registry import CapabilityRegistry


@dataclass
class CapabilityResolution:
    """Result of resolving a set of required capabilities against what is installed.

    all_satisfied: True only when every required capability is currently registered.
    missing: capabilities not found in the registry.
    unsupported: subset of missing for which no marketplace extension exists.
    suggestions: extensions that can satisfy the installable-missing capabilities.
    """

    all_satisfied: bool
    missing: list[str]
    unsupported: list[str]
    suggestions: list[InstallSuggestion]


class CapabilityManager:
    """Resolves required capabilities against the extension marketplace.

    The planner calls resolve() to determine whether a goal can be executed
    immediately or requires marketplace extensions to be installed first.

    CapabilityManager never installs packages or mutates registries — those
    responsibilities belong to PackageInstaller. It only reads the current
    state of the CapabilityRegistry (updated in-place by PackageInstaller
    and restored at startup via restore_from_db()).
    """

    def __init__(
        self,
        catalog: ExtensionCatalog,
        capability_registry: CapabilityRegistry,
    ) -> None:
        self._catalog = catalog
        self._capability_registry = capability_registry

    def resolve(self, required_capabilities: list[str]) -> CapabilityResolution:
        """Check which required capabilities are available and build install suggestions."""
        missing = self.missing_capabilities(required_capabilities)
        if not missing:
            return CapabilityResolution(
                all_satisfied=True,
                missing=[],
                unsupported=[],
                suggestions=[],
            )

        unsupported = [c for c in missing if not self._catalog.find_by_capability(c)]
        installable = [c for c in missing if self._catalog.find_by_capability(c)]
        suggestions = self.install_suggestions(installable)

        return CapabilityResolution(
            all_satisfied=False,
            missing=missing,
            unsupported=unsupported,
            suggestions=suggestions,
        )

    def installed_capabilities(self) -> list[str]:
        """All capabilities currently registered in the platform."""
        return list(self._capability_registry.all_capabilities())

    def missing_capabilities(self, required: list[str]) -> list[str]:
        """Capabilities from required that are not currently installed."""
        installed = set(self.installed_capabilities())
        return [c for c in required if c not in installed]

    def catalog_capabilities(self) -> list[str]:
        """All capability names declared in any catalog package (installed or not).

        Used to extend the GoalAnalyzer allow-list so the LLM can request capabilities
        that exist in the marketplace but are not yet installed. Without this, the
        GoalAnalyzer would never request an uninstalled capability, making the
        pending_install flow unreachable.
        """
        seen: set[str] = set()
        result: list[str] = []
        for pkg in self._catalog.all():
            for cap in pkg.capabilities:
                if cap not in seen:
                    seen.add(cap)
                    result.append(cap)
        return result

    def catalog_capability_descriptions(self) -> dict[str, str]:
        """Map of capability name → human-readable description for all catalog packages.

        Used to inject marketplace capability descriptions into the GoalAnalyzer prompt
        so the LLM can recognize goals that require not-yet-installed capabilities.
        Without descriptions, the LLM sees a capability name like 'filesystem_read' but
        has no context to match it to a goal like 'Read README.md'.
        """
        result: dict[str, str] = {}
        for pkg in self._catalog.all():
            # Use the package description as the capability description.
            # If a package provides multiple capabilities they all share the description.
            desc = pkg.description
            for cap in pkg.capabilities:
                if cap not in result:
                    result[cap] = desc
        return result

    def install_suggestions(self, missing_capabilities: list[str]) -> list[InstallSuggestion]:
        """Find marketplace extensions that provide the given missing capabilities.

        Deduplicates: a package is suggested at most once even if it provides
        multiple missing capabilities.
        """
        seen_ids: set[str] = set()
        suggestions: list[InstallSuggestion] = []

        for cap in missing_capabilities:
            for pkg in self._catalog.find_by_capability(cap):
                if pkg.id not in seen_ids:
                    seen_ids.add(pkg.id)
                    suggestions.append(InstallSuggestion(
                        extension_id=pkg.id,
                        name=pkg.name,
                        description=pkg.description,
                        capabilities_provided=list(pkg.capabilities),
                        permissions=[
                            PermissionSummary(id=p.id, risk_level=p.risk_level)
                            for p in pkg.permissions
                        ],
                    ))

        return suggestions
