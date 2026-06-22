"""Human-in-the-loop approval endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_hitl_manager, get_run_manager
from api.schemas.run import ApprovalRequest, RejectionRequest
from platform.core.exceptions import HITLRejected
from platform.hitl.approval_manager import ApprovalManager
from platform.orchestrator.run_manager import RunManager

router = APIRouter()


@router.post("/{run_id}/approve")
async def approve_run(
    run_id: str,
    request: ApprovalRequest,
    hitl_manager: ApprovalManager = Depends(get_hitl_manager),
    run_manager: RunManager = Depends(get_run_manager),
):
    run_manager.get_run(run_id)  # raises RunNotFound → 404 if run does not exist
    hitl_manager.approve(run_id, request.comment)
    return {"status": "approved", "run_id": run_id}


@router.post("/{run_id}/reject")
async def reject_run(
    run_id: str,
    request: RejectionRequest,
    hitl_manager: ApprovalManager = Depends(get_hitl_manager),
    run_manager: RunManager = Depends(get_run_manager),
):
    run_manager.get_run(run_id)  # raises RunNotFound → 404 if run does not exist
    try:
        hitl_manager.reject(run_id, request.reason)
    except HITLRejected:
        pass  # rejection recorded in run_manager; exception is expected from this side
    return {"status": "rejected", "run_id": run_id}
