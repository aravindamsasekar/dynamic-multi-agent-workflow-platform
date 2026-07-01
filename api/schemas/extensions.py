"""Pydantic schemas for the /extensions API — uses Extension terminology externally."""

from __future__ import annotations

from pydantic import BaseModel


class PackagePermissionResponse(BaseModel):
    id: str
    display_name: str
    description: str
    risk_level: str


class ExtensionResponse(BaseModel):
    id: str
    name: str
    version: str
    description: str
    category: str
    capabilities: list[str]
    tool_names: list[str]
    provides: list[str]
    permissions: list[PackagePermissionResponse]
    installed: bool


class ExtensionListResponse(BaseModel):
    extensions: list[ExtensionResponse]


class InstalledExtensionResponse(BaseModel):
    id: str
    name: str
    version: str
    installed_at: str           # ISO-8601 UTC
    auto_installed: bool
    capabilities_active: list[str]
    permissions_granted: list[str]


class InstalledExtensionListResponse(BaseModel):
    extensions: list[InstalledExtensionResponse]


class InstallExtensionRequest(BaseModel):
    extension_id: str
    permissions_granted: list[str]


class InstallExtensionResponse(BaseModel):
    extension_id: str
    name: str
    version: str
    capabilities_added: list[str]
    tools_added: list[str]
