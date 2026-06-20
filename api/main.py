"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Dynamic Multi-Agent Workflow Platform",
    description="A reusable multi-agent workflow platform with clean architecture.",
    version="0.1.0",
)

# TODO: include routers
# app.include_router(workflows_router, prefix="/workflows", tags=["workflows"])
# app.include_router(runs_router, prefix="/runs", tags=["runs"])
# app.include_router(hitl_router, prefix="/runs", tags=["hitl"])

# TODO: @app.on_event("startup") — load config and populate registries
# TODO: @app.on_event("shutdown") — cleanup resources
