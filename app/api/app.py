"""Minimal FastAPI surface for the Self-Healing Pipeline Agent.

Endpoints:
  GET  /health                          -> liveness + provider/backend info
  GET  /pipelines                       -> registered pipeline definitions
  POST /pipelines/{pipeline_id}/triage  -> run the bounded agent loop on the
                                           pipeline's source data and return the
                                           resulting triage summary
  GET  /runs/{run_id}                   -> full persisted triage state (resumable)

Design notes:
- The agent loop is bounded (iterations, tool calls, timeout) and sub-second on
  demo-scale data, so triage executes within the request and the state is
  persisted for later inspection via ``GET /runs/{run_id}``.
- The LLM provider and storage backend come from settings
  (``SHP_LLM_PROVIDER``, ``SHP_STORAGE_BACKEND``), so the API runs fully
  offline in deterministic mode.
- The app is built via :func:`create_app` so tests can inject overrides.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.llm import create_llm
from app.agent.orchestrator import Orchestrator, TriageConfig
from app.config import settings
from app.data.lineage import LineageEdge, LineageGraph
from app.data.loader import load_source
from app.models import PipelineTriageState
from app.pipeline.registry import PipelineRegistry
from app.storage import store_from_settings
from app.tools.context import make_context


class TriageRequest(BaseModel):
    """Optional overrides for a triage run."""

    source: str | None = Field(
        default=None, description="Source key; defaults to the pipeline's own source."
    )
    limit: int | None = Field(
        default=None, ge=1, description="Cap the number of ingested rows."
    )


class TriageSummary(BaseModel):
    run_id: str
    pipeline_id: str
    phase: str
    deployment_status: str
    final_result: str | None
    detected_issues: list[dict[str, Any]]
    quality_anomalies: list[dict[str, Any]]
    proposed_fix: dict[str, Any] | None
    canary_metrics: dict[str, Any]
    error: str | None


class AppState:
    """Process-wide singletons created during startup."""

    store = None
    registry: PipelineRegistry | None = None
    ctx = None


def _build_context(registry: PipelineRegistry, store):
    pipelines = {p.id: p for p in registry.list()}
    lineage = LineageGraph(
        [
            LineageEdge("orders", "clean.orders", ["order_id", "amount", "currency"]),
            LineageEdge("clean.orders", "daily_revenue", ["amount"]),
        ]
    )
    references = {}
    try:
        customers = load_source("fixtures:customers.csv", str(settings.fixture_dir))
        references["customer_id"] = customers
    except FileNotFoundError:
        pass
    return make_context(
        store=store, pipelines=pipelines, lineage=lineage, references=references
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        AppState.store = store_from_settings(settings)
        AppState.registry = PipelineRegistry(str(settings.pipeline_dir))
        AppState.ctx = _build_context(AppState.registry, AppState.store)
        yield

    app = FastAPI(
        title="Self-Healing Pipeline Agent",
        version="0.1.0",
        description="Observe, diagnose, propose, validate, stage, canary, rollback.",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider.value,
            "storage_backend": settings.storage_backend.value,
            "approval_mode": settings.approval_mode.value,
        }

    @app.get("/pipelines")
    async def list_pipelines() -> list[dict[str, Any]]:
        return [
            {
                "id": p.id,
                "name": p.name,
                "version": p.version,
                "source": p.source,
                "stages": [s.id for s in p.stages],
            }
            for p in AppState.registry.list()
        ]

    @app.post("/pipelines/{pipeline_id}/triage", response_model=TriageSummary)
    async def run_triage(pipeline_id: str, body: TriageRequest | None = None) -> TriageSummary:
        pipeline = AppState.registry.get(pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail=f"unknown pipeline {pipeline_id!r}")

        body = body or TriageRequest()
        source = body.source or pipeline.source
        try:
            df = load_source(source, str(settings.fixture_dir))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if body.limit:
            df = df.head(body.limit)

        llm = create_llm(settings.llm_provider, settings)
        orch = Orchestrator(
            ctx=AppState.ctx,
            llm=llm,
            config=TriageConfig(
                max_iterations=settings.max_iterations,
                max_tool_calls=settings.max_tool_calls,
                timeout_seconds=settings.execution_timeout_seconds,
                max_budget_rows=settings.max_budget_rows,
                approval_mode=settings.approval_mode.value,
            ),
        )
        state = await orch.run(pipeline_id, df)
        return _to_summary(state)

    @app.get("/runs/{run_id}", response_model=PipelineTriageState)
    async def get_run(run_id: str) -> PipelineTriageState:
        state = await AppState.store.metadata.load_state(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return state

    return app


def _to_summary(state: PipelineTriageState) -> TriageSummary:
    return TriageSummary(
        run_id=state.run_id,
        pipeline_id=state.pipeline_id,
        phase=state.phase.value,
        deployment_status=state.deployment_status.value,
        final_result=state.final_result,
        detected_issues=[i.model_dump() for i in state.detected_issues],
        quality_anomalies=[a.model_dump() for a in state.quality_anomalies],
        proposed_fix=state.proposed_fix.model_dump() if state.proposed_fix else None,
        canary_metrics=state.canary_metrics,
        error=state.error,
    )


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.app:app", host="0.0.0.0", port=8000, reload=False)
