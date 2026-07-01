"""InstalledExtensionStore — persistence for marketplace-installed extensions."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from platform.persistence.models import InstallHistoryRow, InstalledPackageRow


class InstalledExtensionStore:
    """Read/write access to installed_packages and install_history tables.

    Uses backend terminology (Package) internally; the API layer translates
    to Extension terminology for callers.
    """

    def list_active(self, session: Session) -> list[InstalledPackageRow]:
        stmt = (
            select(InstalledPackageRow)
            .where(InstalledPackageRow.status == "active")
            .order_by(InstalledPackageRow.installed_at.asc())
        )
        return list(session.scalars(stmt).all())

    def is_installed(self, session: Session, package_id: str) -> bool:
        row = session.get(InstalledPackageRow, package_id)
        return row is not None and row.status == "active"

    def insert(
        self,
        session: Session,
        package_id: str,
        version: str,
        permissions_granted: list[str],
        auto_installed: bool = False,
    ) -> InstalledPackageRow:
        row = InstalledPackageRow(
            id=package_id,
            version=version,
            installed_at=datetime.utcnow(),
            auto_installed=1 if auto_installed else 0,
            permissions_granted=json.dumps(permissions_granted),
            status="active",
        )
        session.add(row)
        session.flush()
        return row

    def record_history(
        self,
        session: Session,
        package_id: str,
        action: str,
        permissions: list[str] | None = None,
        error: str | None = None,
    ) -> InstallHistoryRow:
        row = InstallHistoryRow(
            package_id=package_id,
            action=action,
            timestamp=datetime.utcnow(),
            permissions=json.dumps(permissions) if permissions is not None else None,
            error=error,
        )
        session.add(row)
        session.flush()
        return row
