"""Unit tests for ExtensionCatalog — manifest loading, validation, and queries."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform.extensions.catalog import ExtensionCatalog, ExtensionCatalogError
from platform.extensions.models import CapabilityPackage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_MANIFEST = """\
id: test-pkg
name: Test Package
version: "1.0.0"
description: A test package.
category: filesystem
provides:
  - runtime_agent
dependencies: []
capabilities:
  - test_cap
tools: []
agents: []
agent_prompts: []
permissions:
  - id: test_perm
    display_name: Test Permission
    description: Does something.
    risk_level: low
"""

_TOOL_MANIFEST = """\
id: tool-pkg
name: Tool Package
version: "2.0.0"
description: Package with a tool.
category: filesystem
provides:
  - runtime_agent
dependencies: []
capabilities:
  - read_files
tools:
  - name: read_file_tool
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
  - id: read_perm
    display_name: Read Permission
    description: Allows reading.
    risk_level: medium
"""

_AGENT_PROMPT_MANIFEST = """\
id: prompt-pkg
name: Prompt Package
version: "1.0.0"
description: Package with agent prompts.
category: developer
provides:
  - runtime_agent
dependencies: []
capabilities:
  - do_analysis
tools: []
agents: []
agent_prompts:
  - capability: do_analysis
    system_prompt: |
      You analyze things carefully.
permissions:
  - id: analysis_perm
    display_name: Analysis Permission
    description: Perform analysis.
    risk_level: high
"""


def _write(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TestExtensionCatalogLoad
# ---------------------------------------------------------------------------


class TestExtensionCatalogLoad:
    def test_load_single_manifest(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        catalog = ExtensionCatalog.load(tmp_path)
        assert len(catalog.all()) == 1

    def test_load_multiple_manifests(self, tmp_path: Path) -> None:
        _write(tmp_path, "pkg-a.yaml", _MINIMAL_MANIFEST)
        _write(tmp_path, "pkg-b.yaml", _TOOL_MANIFEST)
        catalog = ExtensionCatalog.load(tmp_path)
        assert len(catalog.all()) == 2

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        catalog = ExtensionCatalog.load(tmp_path)
        assert catalog.all() == []

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, "bad.yaml", "key: [unclosed bracket")
        with pytest.raises(ExtensionCatalogError, match="Failed to parse"):
            ExtensionCatalog.load(tmp_path)

    def test_load_missing_required_field_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, "incomplete.yaml", "name: No ID Here\nversion: '1.0.0'\n")
        with pytest.raises(ExtensionCatalogError, match="missing required fields"):
            ExtensionCatalog.load(tmp_path)

    def test_load_duplicate_id_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, "first.yaml", _MINIMAL_MANIFEST)
        _write(tmp_path, "second.yaml", _MINIMAL_MANIFEST)  # same id: test-pkg
        with pytest.raises(ExtensionCatalogError, match="Duplicate package id"):
            ExtensionCatalog.load(tmp_path)

    def test_load_invalid_risk_level_raises(self, tmp_path: Path) -> None:
        bad = _MINIMAL_MANIFEST.replace("risk_level: low", "risk_level: critical")
        _write(tmp_path, "bad-risk.yaml", bad)
        with pytest.raises(ExtensionCatalogError, match="invalid risk_level"):
            ExtensionCatalog.load(tmp_path)


# ---------------------------------------------------------------------------
# TestExtensionCatalogQueries
# ---------------------------------------------------------------------------


class TestExtensionCatalogQueries:
    def test_get_returns_package(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        catalog = ExtensionCatalog.load(tmp_path)
        pkg = catalog.get("test-pkg")
        assert pkg is not None
        assert isinstance(pkg, CapabilityPackage)
        assert pkg.id == "test-pkg"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        catalog = ExtensionCatalog.load(tmp_path)
        assert catalog.get("nonexistent") is None

    def test_all_returns_list_with_correct_length(self, tmp_path: Path) -> None:
        _write(tmp_path, "pkg-a.yaml", _MINIMAL_MANIFEST)
        _write(tmp_path, "pkg-b.yaml", _TOOL_MANIFEST)
        catalog = ExtensionCatalog.load(tmp_path)
        assert len(catalog.all()) == 2

    def test_find_by_capability_returns_matching(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        catalog = ExtensionCatalog.load(tmp_path)
        results = catalog.find_by_capability("test_cap")
        assert len(results) == 1
        assert results[0].id == "test-pkg"

    def test_find_by_capability_not_found(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        catalog = ExtensionCatalog.load(tmp_path)
        assert catalog.find_by_capability("nonexistent_cap") == []

    def test_find_by_capability_multiple_packages(self, tmp_path: Path) -> None:
        shared_cap = _MINIMAL_MANIFEST
        other = _MINIMAL_MANIFEST.replace("id: test-pkg", "id: other-pkg")
        _write(tmp_path, "first.yaml", shared_cap)
        _write(tmp_path, "second.yaml", other)
        catalog = ExtensionCatalog.load(tmp_path)
        results = catalog.find_by_capability("test_cap")
        assert len(results) == 2
        found_ids = {p.id for p in results}
        assert found_ids == {"test-pkg", "other-pkg"}


# ---------------------------------------------------------------------------
# TestExtensionCatalogParsedFields
# ---------------------------------------------------------------------------


class TestExtensionCatalogParsedFields:
    def test_parses_top_level_fields(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        pkg = ExtensionCatalog.load(tmp_path).get("test-pkg")
        assert pkg.id == "test-pkg"
        assert pkg.name == "Test Package"
        assert pkg.version == "1.0.0"
        assert pkg.description == "A test package."
        assert pkg.category == "filesystem"
        assert pkg.provides == ["runtime_agent"]
        assert pkg.dependencies == []
        assert pkg.capabilities == ["test_cap"]

    def test_parses_tool_manifest(self, tmp_path: Path) -> None:
        _write(tmp_path, "tool-pkg.yaml", _TOOL_MANIFEST)
        pkg = ExtensionCatalog.load(tmp_path).get("tool-pkg")
        assert len(pkg.tools) == 1
        tool = pkg.tools[0]
        assert tool.name == "read_file_tool"
        assert tool.description == "Reads a file."
        assert tool.adapter_type == "filesystem"
        assert tool.adapter_config == {}
        assert "path" in tool.input_schema["properties"]

    def test_parses_agent_prompt_template(self, tmp_path: Path) -> None:
        _write(tmp_path, "prompt-pkg.yaml", _AGENT_PROMPT_MANIFEST)
        pkg = ExtensionCatalog.load(tmp_path).get("prompt-pkg")
        assert len(pkg.agent_prompts) == 1
        ap = pkg.agent_prompts[0]
        assert ap.capability == "do_analysis"
        assert "analyze" in ap.system_prompt.lower()

    def test_parses_permission_fields(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        pkg = ExtensionCatalog.load(tmp_path).get("test-pkg")
        assert len(pkg.permissions) == 1
        perm = pkg.permissions[0]
        assert perm.id == "test_perm"
        assert perm.display_name == "Test Permission"
        assert perm.description == "Does something."
        assert perm.risk_level == "low"

    def test_empty_agents_list_ok(self, tmp_path: Path) -> None:
        _write(tmp_path, "test-pkg.yaml", _MINIMAL_MANIFEST)
        pkg = ExtensionCatalog.load(tmp_path).get("test-pkg")
        assert pkg.agents == []
