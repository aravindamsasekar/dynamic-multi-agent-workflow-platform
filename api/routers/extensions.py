"""Extensions API — browse and install capability marketplace extensions."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import (
    get_db_session,
    get_extension_catalog,
    get_installed_extension_store,
    get_package_installer,
)
from api.schemas.extensions import (
    ExtensionListResponse,
    ExtensionResponse,
    InstallExtensionRequest,
    InstallExtensionResponse,
    InstalledExtensionListResponse,
    InstalledExtensionResponse,
    PackagePermissionResponse,
)
from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.installer import (
    DependencyNotInstalledError,
    ExtensionAlreadyInstalledError,
    ExtensionNotFoundError,
    MissingPermissionError,
    PackageInstaller,
)
from platform.extensions.models import CapabilityPackage
from platform.persistence.models import InstalledPackageRow
from platform.persistence.repositories.package_repo import InstalledExtensionStore

router = APIRouter()


def _to_extension_response(pkg: CapabilityPackage, *, installed: bool) -> ExtensionResponse:
    return ExtensionResponse(
        id=pkg.id,
        name=pkg.name,
        version=pkg.version,
        description=pkg.description,
        category=pkg.category,
        capabilities=pkg.capabilities,
        provides=pkg.provides,
        permissions=[
            PackagePermissionResponse(
                id=p.id,
                display_name=p.display_name,
                description=p.description,
                risk_level=p.risk_level,
            )
            for p in pkg.permissions
        ],
        installed=installed,
    )


def _to_installed_response(
    row: InstalledPackageRow, pkg: CapabilityPackage
) -> InstalledExtensionResponse:
    permissions = json.loads(row.permissions_granted) if row.permissions_granted else []
    return InstalledExtensionResponse(
        id=row.id,
        name=pkg.name,
        version=row.version,
        installed_at=row.installed_at.isoformat() + "Z",
        capabilities_active=list(pkg.capabilities),
        permissions_granted=permissions,
    )


@router.get("", response_model=ExtensionListResponse)
def list_extensions(
    catalog: ExtensionCatalog = Depends(get_extension_catalog),
    store: InstalledExtensionStore = Depends(get_installed_extension_store),
    session: Session = Depends(get_db_session),
) -> ExtensionListResponse:
    """List all extensions in the catalog with their current installation status."""
    installed_ids = {row.id for row in store.list_active(session)}
    return ExtensionListResponse(
        extensions=[
            _to_extension_response(pkg, installed=pkg.id in installed_ids)
            for pkg in catalog.all()
        ]
    )


@router.get("/installed", response_model=InstalledExtensionListResponse)
def list_installed_extensions(
    catalog: ExtensionCatalog = Depends(get_extension_catalog),
    store: InstalledExtensionStore = Depends(get_installed_extension_store),
    session: Session = Depends(get_db_session),
) -> InstalledExtensionListResponse:
    """List extensions installed via the marketplace."""
    rows = store.list_active(session)
    extensions = []
    for row in rows:
        pkg = catalog.get(row.id)
        if pkg is not None:
            extensions.append(_to_installed_response(row, pkg))
    return InstalledExtensionListResponse(extensions=extensions)


@router.post("/install", response_model=InstallExtensionResponse, status_code=201)
def install_extension(
    request: InstallExtensionRequest,
    installer: PackageInstaller = Depends(get_package_installer),
    session: Session = Depends(get_db_session),
) -> InstallExtensionResponse:
    """Install an extension from the marketplace catalog."""
    try:
        result = installer.install(
            extension_id=request.extension_id,
            permissions_granted=request.permissions_granted,
            session=session,
        )
        session.commit()
    except ExtensionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ExtensionAlreadyInstalledError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except MissingPermissionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DependencyNotInstalledError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return InstallExtensionResponse(
        extension_id=result.extension_id,
        name=result.name,
        version=result.version,
        capabilities_added=result.capabilities_added,
        tools_added=result.tools_added,
    )
