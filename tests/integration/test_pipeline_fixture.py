"""Integration tests: full pipeline on real fixtures + shadow validation + rollback."""

from __future__ import annotations

import polars as pl
import pytest

from app.data.loader import load_source
from app.data.lineage import LineageEdge, LineageGraph
from app.models.state import Phase
from app.pipeline.registry import PipelineRegistry
from app.pipeline.runner import run_pipeline
from app.rollback.manager import RollbackManager
from app.storage import MemoryMetadataStore, MemoryWarehouse, BackendStore
from app.tools.context import ToolContext
from app.agent.llm import DeterministicLLM
from app.agent.orchestrator import Orchestrator, TriageConfig
from app.validation.engine import shadow_validate


def _orders_pipeline():
    return PipelineRegistry("./pipelines").get("orders")


def _healthy_df():
    return load_source("fixtures:orders.csv", "./datasets/fixtures")


def _ctx(pipeline):
    store = BackendStore(MemoryMetadataStore(), MemoryWarehouse())
    lineage = LineageGraph(
        [
            LineageEdge("orders", "clean.orders", ["order_id", "amount", "currency"]),
            LineageEdge("clean.orders", "daily_revenue", ["amount"]),
        ]
    )
    return ToolContext(store=store, pipelines={pipeline.id: pipeline}, lineage=lineage)


class TestHealthyFixture:
    @pytest.mark.asyncio
    async def test_healthy_data_is_not_mutated(self):
        pipeline = _orders_pipeline()
        ctx = _ctx(pipeline)
        orch = Orchestrator(ctx=ctx, llm=DeterministicLLM(), config=TriageConfig())
        state = await orch.run(pipeline.id, _healthy_df())
        # No drift/anomaly -> no destructive action, no deployment.
        assert state.detected_issues == []
        assert state.phase == Phase.SAFE_STOP
        assert state.deployment_status.value == "not_deployed"

    @pytest.mark.asyncio
    async def test_runner_persists_to_shadow_namespace(self):
        pipeline = _orders_pipeline()
        store = BackendStore(MemoryMetadataStore(), MemoryWarehouse())
        df = _healthy_df()
        res = await run_pipeline(pipeline, df, store.warehouse, namespace="shadow")
        assert res.ok is True
        versions = await store.warehouse.list_versions("daily_revenue")
        assert versions == ["shadow:v1.0"]


class TestShadowValidationConstraints:
    @pytest.mark.asyncio
    async def test_mass_data_loss_blocked(self):
        pipeline = _orders_pipeline()
        store = BackendStore(MemoryMetadataStore(), MemoryWarehouse())
        df = _healthy_df()
        current = await run_pipeline(pipeline, df, store.warehouse, persist=False)
        # Candidate that drops 90% of rows.
        cand_df = df.head(df.height // 10)
        candidate = await run_pipeline(pipeline, cand_df, store.warehouse, persist=False)
        report = shadow_validate(
            current, candidate,
            pipeline_id="orders",
            expected_schema=pipeline.stages[-1].output_schema,
            input_row_count=df.height,
        )
        assert report.passed is False
        names = [c.name for c in report.failed]
        assert any("row_count" in n for n in names)

    @pytest.mark.asyncio
    async def test_silent_corruption_blocked_by_metric_delta(self):
        pipeline = _orders_pipeline()
        store = BackendStore(MemoryMetadataStore(), MemoryWarehouse())
        df = _healthy_df()
        current = await run_pipeline(pipeline, df, store.warehouse, persist=False)
        # Candidate that preserves row count but halves all amounts (corruption).
        corrupted = df.with_columns((pl.col("amount") * 0.5).alias("amount"))
        candidate = await run_pipeline(pipeline, corrupted, store.warehouse, persist=False)
        report = shadow_validate(
            current, candidate,
            pipeline_id="orders",
            expected_schema=pipeline.stages[-1].output_schema,
            input_row_count=df.height,
        )
        assert report.passed is False
        aggregates = [c.name for c in report.failed if "aggregate" in c.name]
        assert aggregates

    @pytest.mark.asyncio
    async def test_constraint_violation_blocked(self):
        from app.pipeline.runner import RunResult, StageResult

        # Duplicate order_id in a row-preserving output -> uniqueness fails.
        df = pl.DataFrame(
            {
                "order_id": [1, 1, 2],  # duplicate PK
                "amount": [10.0, 20.0, 30.0],
                "currency": ["USD", "USD", "USD"],
                "created_at": ["2026-01-01"] * 3,
                "status": ["a", "b", "c"],
                "customer_id": [1, 2, 3],
            }
        )
        result = RunResult(
            pipeline_id="orders", ok=True,
            final_df=df,
            stages=[StageResult(stage_id="x", ok=True)],
        )
        report = shadow_validate(
            current=result,
            candidate=result,
            pipeline_id="orders",
            input_row_count=df.height,
            constraints={"unique": ["order_id"]},
        )
        assert report.passed is False
        assert any(c.name == "unique.order_id" for c in report.failed)


class TestRollbackIdempotency:
    @pytest.mark.asyncio
    async def test_rollback_restores_previous_version(self):
        store = BackendStore(MemoryMetadataStore(), MemoryWarehouse())
        await store.warehouse.write_table("t", "main:v1", pl.DataFrame({"a": [1, 2, 3]}))
        await store.warehouse.write_table("t", "main:v2", pl.DataFrame({"a": [9, 9]}))

        rm = RollbackManager(store.warehouse, store.metadata)
        rm.set_active("t", "v1")
        rm.set_active("t", "v2")

        report = await rm.rollback("run1", "t", reason="canary failed")
        assert report.rolled_back is True
        assert report.previous_version == "v2"
        assert report.restored_version == "v1"
        restored = await store.warehouse.read_table("t", "main:v1")
        assert restored is not None and restored.height == 3
        # Rollback is idempotent: second call is a no-op.
        report2 = await rm.rollback("run1", "t", reason="again")
        assert report2.rolled_back is False
        # Audit record persisted.
        recs = await store.metadata.get_rollback_records(run_id="run1")
        assert len(recs) == 1
        assert recs[0]["restored_version"] == "v1"
