"""Unit tests for CapabilityManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.manager import CapabilityManager, CapabilityResolution
from platform.extensions.models import InstallSuggestion
from platform.planner.capability_registry import CapabilityRegistry

_EXTENSIONS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "resources" / "extensions"
)

_CATALOG = ExtensionCatalog.load(_EXTENSIONS_DIR)


def _empty_registry() -> CapabilityRegistry:
    return CapabilityRegistry()


def _registry_with(*capabilities: str) -> CapabilityRegistry:
    """Create a CapabilityRegistry with the given generatable capabilities registered."""
    reg = CapabilityRegistry()
    for cap in capabilities:
        reg.register_generatable_capability(cap)
    return reg


# ---------------------------------------------------------------------------
# CapabilityManager.resolve()
# ---------------------------------------------------------------------------


class TestResolve:
    def test_all_satisfied_when_empty_required(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        result = mgr.resolve([])
        assert result.all_satisfied is True
        assert result.missing == []
        assert result.unsupported == []
        assert result.suggestions == []

    def test_all_satisfied_when_all_installed(self) -> None:
        reg = _registry_with("filesystem_read")
        mgr = CapabilityManager(_CATALOG, reg)
        result = mgr.resolve(["filesystem_read"])
        assert result.all_satisfied is True

    def test_missing_when_capability_not_installed(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        result = mgr.resolve(["filesystem_read"])
        assert result.all_satisfied is False
        assert "filesystem_read" in result.missing

    def test_installable_capability_not_in_unsupported(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        result = mgr.resolve(["filesystem_read"])
        assert "filesystem_read" not in result.unsupported

    def test_unsupported_capability_when_not_in_catalog(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        result = mgr.resolve(["totally_unknown_cap"])
        assert result.all_satisfied is False
        assert "totally_unknown_cap" in result.unsupported
        assert result.suggestions == []

    def test_mixed_installable_and_unsupported(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        result = mgr.resolve(["filesystem_read", "mystery_cap"])
        assert result.all_satisfied is False
        assert "filesystem_read" in result.missing
        assert "mystery_cap" in result.missing
        assert "mystery_cap" in result.unsupported
        assert "filesystem_read" not in result.unsupported

    def test_suggestions_populated_for_installable_missing(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        result = mgr.resolve(["filesystem_read"])
        assert len(result.suggestions) >= 1
        ids = [s.extension_id for s in result.suggestions]
        assert "filesystem-reader" in ids

    def test_partial_satisfaction_leaves_remaining_missing(self) -> None:
        reg = _registry_with("filesystem_read")
        mgr = CapabilityManager(_CATALOG, reg)
        result = mgr.resolve(["filesystem_read", "mystery_cap"])
        assert result.all_satisfied is False
        assert "mystery_cap" in result.missing
        assert "filesystem_read" not in result.missing


# ---------------------------------------------------------------------------
# CapabilityManager.installed_capabilities()
# ---------------------------------------------------------------------------


class TestInstalledCapabilities:
    def test_empty_registry_returns_empty(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        assert mgr.installed_capabilities() == []

    def test_returns_registered_generatable_capabilities(self) -> None:
        reg = _registry_with("filesystem_read", "some_other")
        mgr = CapabilityManager(_CATALOG, reg)
        caps = mgr.installed_capabilities()
        assert "filesystem_read" in caps
        assert "some_other" in caps

    def test_returns_generatable_and_agent_capabilities(self) -> None:
        # installed_capabilities() reflects all_capabilities(): agent caps + generatable caps.
        # Raw tool capabilities (ToolCapabilityDescriptor) are not in all_capabilities()
        # because the GoalAnalyzer requests agent-level capabilities, not tool-level ones.
        reg = _registry_with("some_generatable_cap")
        mgr = CapabilityManager(_CATALOG, reg)
        assert "some_generatable_cap" in mgr.installed_capabilities()


# ---------------------------------------------------------------------------
# CapabilityManager.missing_capabilities()
# ---------------------------------------------------------------------------


class TestMissingCapabilities:
    def test_empty_required_gives_no_missing(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        assert mgr.missing_capabilities([]) == []

    def test_uninstalled_capability_is_missing(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        assert "filesystem_read" in mgr.missing_capabilities(["filesystem_read"])

    def test_installed_capability_not_missing(self) -> None:
        reg = _registry_with("filesystem_read")
        mgr = CapabilityManager(_CATALOG, reg)
        assert mgr.missing_capabilities(["filesystem_read"]) == []


# ---------------------------------------------------------------------------
# CapabilityManager.install_suggestions()
# ---------------------------------------------------------------------------


class TestInstallSuggestions:
    def test_returns_suggestion_for_catalog_capability(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        suggestions = mgr.install_suggestions(["filesystem_read"])
        assert len(suggestions) >= 1
        assert all(isinstance(s, InstallSuggestion) for s in suggestions)

    def test_returns_empty_for_unknown_capability(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        assert mgr.install_suggestions(["no_such_cap"]) == []

    def test_deduplicates_when_package_covers_multiple_missing(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        # filesystem-reader provides filesystem_read; suggest it only once
        # even if we ask about the same cap twice (simulates two missing caps
        # served by the same package)
        suggestions = mgr.install_suggestions(["filesystem_read", "filesystem_read"])
        ids = [s.extension_id for s in suggestions]
        assert ids.count("filesystem-reader") == 1

    def test_suggestion_fields_populated(self) -> None:
        mgr = CapabilityManager(_CATALOG, _empty_registry())
        suggestions = mgr.install_suggestions(["filesystem_read"])
        s = next(s for s in suggestions if s.extension_id == "filesystem-reader")
        assert s.name == "Filesystem Reader"
        assert "filesystem_read" in s.capabilities_provided
        assert len(s.permissions) >= 1
        assert all(p.risk_level in ("low", "medium", "high") for p in s.permissions)
