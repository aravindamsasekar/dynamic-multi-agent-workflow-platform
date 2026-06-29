"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.dependencies import initialize, run_startup_indexing, shutdown
from api.routers import hitl as hitl_router
from api.routers import knowledge as knowledge_router
from api.routers import runs as runs_router
from api.routers import workflows as workflows_router
from platform.core.exceptions import RunNotFound, WorkflowNotFound


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize(Path("workflows"))
    await run_startup_indexing()
    yield
    await shutdown()


app = FastAPI(
    title="Dynamic Multi-Agent Workflow Platform",
    description="A reusable multi-agent workflow platform with clean architecture.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(WorkflowNotFound)
async def _workflow_not_found(request, exc: WorkflowNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(RunNotFound)
async def _run_not_found(request, exc: RunNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


app.include_router(workflows_router.router, prefix="/workflows", tags=["workflows"])
app.include_router(runs_router.router, prefix="/runs", tags=["runs"])
app.include_router(hitl_router.router, prefix="/runs", tags=["hitl"])
app.include_router(knowledge_router.router, prefix="/knowledge", tags=["knowledge"])
