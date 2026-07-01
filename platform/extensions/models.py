"""Data models for the V4 capability marketplace."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PackagePermission:
    id: str
    display_name: str
    description: str
    risk_level: str  # "low" | "medium" | "high"


@dataclass
class ToolManifest:
    name: str
    description: str
    adapter_type: str
    adapter_config: dict
    input_schema: dict


@dataclass
class AgentManifest:
    id: str
    name: str
    description: str
    capabilities: list[str]
    consumes: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)


@dataclass
class AgentPromptTemplate:
    capability: str
    system_prompt: str


@dataclass
class PermissionSummary:
    """Minimal permission description surfaced in install suggestions."""
    id: str
    risk_level: str  # "low" | "medium" | "high"


@dataclass
class InstallSuggestion:
    """A marketplace extension that can satisfy one or more missing capabilities."""
    extension_id: str
    name: str
    description: str
    capabilities_provided: list[str]
    permissions: list[PermissionSummary]


@dataclass
class CapabilityPackage:
    id: str
    name: str
    version: str
    description: str
    category: str            # "filesystem" | "github" | "knowledge" | ...
    provides: list[str]      # ["runtime_agent"] | ["static_agent"] | both
    dependencies: list[str]  # package IDs that must be installed first
    capabilities: list[str]
    tools: list[ToolManifest]
    agents: list[AgentManifest]
    agent_prompts: list[AgentPromptTemplate]
    permissions: list[PackagePermission]
