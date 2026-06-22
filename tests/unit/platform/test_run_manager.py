"""Unit tests for RunManager."""

from __future__ import annotations

import asyncio

import pytest

from platform.core.exceptions import RunNotFound
from platform.core.models.workflow import RunStatus
from platform.orchestrator.run_manager import RunManager


class TestRunManager:
    # ------------------------------------------------------------------
    # create_run
    # ------------------------------------------------------------------

    def test_create_run_returns_pending_status(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "hello")
        assert run.status == RunStatus.PENDING

    def test_create_run_stores_workflow_id_and_input(self):
        manager = RunManager()
        run = manager.create_run("wf-42", "my input")
        assert run.workflow_id == "wf-42"
        assert run.input == "my input"

    def test_create_run_generates_unique_run_ids(self):
        manager = RunManager()
        r1 = manager.create_run("wf-1", "a")
        r2 = manager.create_run("wf-1", "b")
        assert r1.run_id != r2.run_id

    def test_create_run_is_retrievable(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")
        assert manager.get_run(run.run_id) is run

    # ------------------------------------------------------------------
    # get_run
    # ------------------------------------------------------------------

    def test_get_run_returns_correct_run(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")
        fetched = manager.get_run(run.run_id)
        assert fetched.run_id == run.run_id

    def test_get_run_raises_run_not_found(self):
        manager = RunManager()
        with pytest.raises(RunNotFound):
            manager.get_run("nonexistent-id")

    # ------------------------------------------------------------------
    # update_status
    # ------------------------------------------------------------------

    def test_update_status_transitions_status(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")
        manager.update_status(run.run_id, RunStatus.RUNNING)
        assert run.status == RunStatus.RUNNING

    async def test_update_status_bumps_updated_at(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")
        original = run.updated_at
        await asyncio.sleep(0.005)
        manager.update_status(run.run_id, RunStatus.RUNNING)
        assert run.updated_at > original

    # ------------------------------------------------------------------
    # complete / fail
    # ------------------------------------------------------------------

    def test_complete_sets_completed_status_and_output(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")
        manager.complete(run.run_id, "the result")
        assert run.status == RunStatus.COMPLETED
        assert run.output == "the result"

    def test_fail_sets_failed_status_and_error(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")
        manager.fail(run.run_id, "something broke")
        assert run.status == RunStatus.FAILED
        assert run.error == "something broke"

    # ------------------------------------------------------------------
    # list_runs
    # ------------------------------------------------------------------

    def test_list_runs_empty(self):
        manager = RunManager()
        assert manager.list_runs() == []

    def test_list_runs_returns_all(self):
        manager = RunManager()
        r1 = manager.create_run("wf-1", "a")
        r2 = manager.create_run("wf-2", "b")
        runs = manager.list_runs()
        assert len(runs) == 2
        assert r1 in runs
        assert r2 in runs

    # ------------------------------------------------------------------
    # pause / resume — asyncio coordination
    # ------------------------------------------------------------------

    async def test_pause_sets_waiting_approval_before_blocking(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")

        task = asyncio.create_task(manager.pause(run.run_id))
        await asyncio.sleep(0)  # yield so pause() reaches event.wait()

        assert run.status == RunStatus.WAITING_APPROVAL
        assert not task.done()

        manager.resume(run.run_id)
        await task

    async def test_resume_unblocks_paused_run(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")

        pause_completed = False

        async def do_pause() -> None:
            nonlocal pause_completed
            await manager.pause(run.run_id)
            pause_completed = True

        task = asyncio.create_task(do_pause())
        await asyncio.sleep(0)
        assert not pause_completed

        manager.resume(run.run_id)
        await task

        assert pause_completed
        assert run.status == RunStatus.RUNNING

    async def test_fail_unblocks_paused_run(self):
        manager = RunManager()
        run = manager.create_run("wf-1", "input")

        pause_completed = False

        async def do_pause() -> None:
            nonlocal pause_completed
            await manager.pause(run.run_id)
            pause_completed = True

        task = asyncio.create_task(do_pause())
        await asyncio.sleep(0)
        assert not pause_completed

        manager.fail(run.run_id, "rejected")
        await task

        assert pause_completed
        assert run.status == RunStatus.FAILED
        assert run.error == "rejected"
