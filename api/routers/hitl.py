"""Human-in-the-loop approval endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/{run_id}/approve")
async def approve_run(run_id: str):
    # TODO: inject ApprovalManager, call hitl_manager.approve(run_id), return 200
    raise NotImplementedError


@router.post("/{run_id}/reject")
async def reject_run(run_id: str):
    # TODO: inject ApprovalManager, call hitl_manager.reject(run_id), return 200
    raise NotImplementedError
