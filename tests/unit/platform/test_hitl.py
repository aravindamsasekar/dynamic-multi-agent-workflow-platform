"""Unit tests for ApprovalManager."""

from __future__ import annotations

import asyncio

import pytest

from platform.core.exceptions import HITLRejected
from platform.core.models.workflow import RunStatus
from platform.hitl.approval_manager import ApprovalManager
from platform.orchestrator.run_manager import RunManager


def make_manager() -> tuple[RunManager, ApprovalManager]:
    run_manager = RunManager()
    approval = ApprovalManager(run_manager=run_manager)
    return run_manager, approval


class TestApprovalManager:
    # ------------------------------------------------------------------
    # get_pending (query)
    # ------------------------------------------------------------------

    def test_get_pending_returns_none_when_not_pending(self):
        _, approval = make_manager()
        assert approval.get_pending("unknown-run") is None

    # ------------------------------------------------------------------
    # request_approval
    # ------------------------------------------------------------------

    async def test_request_approval_stores_context(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")
        ctx = {"reason": "needs human check", "data": 42}

        task = asyncio.create_task(approval.request_approval(run.run_id, ctx))
        await asyncio.sleep(0)  # let request_approval reach pause → event.wait()

        assert approval.get_pending(run.run_id) == ctx

        run_manager.resume(run.run_id)
        await task

    async def test_request_approval_sets_waiting_approval_status(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        task = asyncio.create_task(approval.request_approval(run.run_id, {}))
        await asyncio.sleep(0)

        assert run.status == RunStatus.WAITING_APPROVAL

        run_manager.resume(run.run_id)
        await task

    # ------------------------------------------------------------------
    # approve
    # ------------------------------------------------------------------

    async def test_approve_resumes_run(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        task = asyncio.create_task(approval.request_approval(run.run_id, {}))
        await asyncio.sleep(0)

        approval.approve(run.run_id)
        await task

        assert run.status == RunStatus.RUNNING

    async def test_approve_clears_pending(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        task = asyncio.create_task(approval.request_approval(run.run_id, {"k": "v"}))
        await asyncio.sleep(0)

        approval.approve(run.run_id)
        await task

        assert approval.get_pending(run.run_id) is None

    async def test_approve_unblocks_request_approval(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        returned = False

        async def request() -> None:
            nonlocal returned
            await approval.request_approval(run.run_id, {})
            returned = True

        task = asyncio.create_task(request())
        await asyncio.sleep(0)
        assert not returned

        approval.approve(run.run_id)
        await task
        assert returned

    # ------------------------------------------------------------------
    # reject
    # ------------------------------------------------------------------

    async def test_reject_fails_run(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        task = asyncio.create_task(approval.request_approval(run.run_id, {}))
        await asyncio.sleep(0)

        with pytest.raises(HITLRejected):
            approval.reject(run.run_id, "not approved")

        await task
        assert run.status == RunStatus.FAILED

    async def test_reject_raises_hitl_rejected(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        task = asyncio.create_task(approval.request_approval(run.run_id, {}))
        await asyncio.sleep(0)

        with pytest.raises(HITLRejected):
            approval.reject(run.run_id, "bad output")

        await task

    async def test_reject_clears_pending(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        task = asyncio.create_task(approval.request_approval(run.run_id, {"k": "v"}))
        await asyncio.sleep(0)

        with pytest.raises(HITLRejected):
            approval.reject(run.run_id, "no")

        await task
        assert approval.get_pending(run.run_id) is None

    # ------------------------------------------------------------------
    # Full end-to-end flow
    # ------------------------------------------------------------------

    async def test_full_approve_flow(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")
        ctx = {"step": "final review"}

        request_returned = False

        async def request() -> None:
            nonlocal request_returned
            await approval.request_approval(run.run_id, ctx)
            request_returned = True

        task = asyncio.create_task(request())
        await asyncio.sleep(0)

        assert run.status == RunStatus.WAITING_APPROVAL
        assert approval.get_pending(run.run_id) == ctx
        assert not request_returned

        approval.approve(run.run_id, comment="looks good")
        await task

        assert request_returned
        assert run.status == RunStatus.RUNNING
        assert approval.get_pending(run.run_id) is None

    async def test_full_reject_flow(self):
        run_manager, approval = make_manager()
        run = run_manager.create_run("wf-1", "input")

        task = asyncio.create_task(approval.request_approval(run.run_id, {"step": "review"}))
        await asyncio.sleep(0)

        assert run.status == RunStatus.WAITING_APPROVAL

        with pytest.raises(HITLRejected):
            approval.reject(run.run_id, reason="output unsafe")

        await task

        assert run.status == RunStatus.FAILED
        assert run.error == "output unsafe"
        assert approval.get_pending(run.run_id) is None
