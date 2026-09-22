"""Tool-layer tests: every registered tool works through the policy-gated registry."""

from __future__ import annotations

import pytest

from app.agent.policies import PolicyEngine, PolicyError
from app.evaluation.runner import _fresh_ctx, build_environment
from app.models.state import ApprovalStatus, Phase
from app.tools.registry import ToolRegistry


@pytest.fixture()
def setup():
    env = build_environment()
    ctx, pipeline = _fresh_ctx(env)
    ctx.observed[pipeline.id] = env.healthy
    registry = ToolRegistry(ctx, PolicyEngine(approval_mode="auto"))
    return ctx, pipeline, registry


class TestSchemaTools:
    @pytest.mark.asyncio
    async def test_expected_and_observed_schema(self, setup):
        _, pipeline, reg = setup
        expected = await reg.invoke("get_expected_schema", Phase.OBSERVE, pipeline_id=pipeline.id)
        observed = await reg.invoke("get_observed_schema", Phase.OBSERVE, pipeline_id=pipeline.id)
        assert expected["table"] == "orders"
        names = {c["name"] for c in observed["columns"]}
        assert {"order_id", "amount", "currency"} <= names

    @pytest.mark.asyncio
    async def test_compare_schemas_healthy_no_drift(self, setup):
        _, pipeline, reg = setup
        out = await reg.invoke("compare_schemas", Phase.OBSERVE, pipeline_id=pipeline.id)
        assert out["drift_events"] == []

    @pytest.mark.asyncio
    async def test_schema_history(self, setup):
        ctx, _, reg = setup
        from app.models import ColumnDefinition, DataType, SchemaDefinition

        await ctx.store.metadata.save_schema(
            SchemaDefinition(table="orders", version=1,
                             columns=[ColumnDefinition(name="a", type=DataType.INTEGER)])
        )
        out = await reg.invoke("get_schema_history", Phase.OBSERVE, table="orders")
        assert out["versions"][0]["version"] == 1


class TestDataTools:
    @pytest.mark.asyncio
    async def test_profile_query_sample(self, setup):
        _, pipeline, reg = setup
        prof = await reg.invoke("profile_dataset", Phase.OBSERVE, pipeline_id=pipeline.id)
        assert prof["row_count"] == 600
        rows = await reg.invoke(
            "query_dataset", Phase.OBSERVE, pipeline_id=pipeline.id,
            columns=["order_id", "amount"], limit=5,
        )
        assert len(rows["order_id"]) == 5
        sample = await reg.invoke(
            "sample_rows", Phase.OBSERVE, pipeline_id=pipeline.id, column="currency", limit=3
        )
        assert len(sample["rows"]["currency"]) == 3

    @pytest.mark.asyncio
    async def test_quality_checks_and_metric_history(self, setup):
        ctx, pipeline, reg = setup
        out = await reg.invoke("run_quality_checks", Phase.OBSERVE, pipeline_id=pipeline.id)
        assert out["passed"] is True
        hist = await reg.invoke("get_metric_history", Phase.OBSERVE, table="orders")
        assert hist["history"] == []

    @pytest.mark.asyncio
    async def test_compare_distributions(self, setup):
        _, pipeline, reg = setup
        out = await reg.invoke(
            "compare_distributions", Phase.OBSERVE, pipeline_id=pipeline.id, column="currency"
        )
        assert out["column"] == "currency"
        assert len(out["value_counts"]) > 0


class TestPipelineTools:
    @pytest.mark.asyncio
    async def test_get_pipeline_and_lineage(self, setup):
        ctx, pipeline, reg = setup
        p = await reg.invoke("get_pipeline", Phase.OBSERVE, pipeline_id=pipeline.id)
        assert p["id"] == pipeline.id
        lineage = await reg.invoke(
            "get_lineage", Phase.OBSERVE, changed_columns=["amount"]
        )
        assert "daily_revenue" in lineage["blast_radius"]
        run = await reg.invoke("get_pipeline_run", Phase.OBSERVE, pipeline_id=pipeline.id)
        assert run["status"] == "no_record"
        logs = await reg.invoke("get_stage_logs", Phase.OBSERVE, pipeline_id=pipeline.id)
        assert logs["pipeline_id"] == pipeline.id

    @pytest.mark.asyncio
    async def test_run_stage_shadow_write_gated(self, setup):
        _, pipeline, reg = setup
        with pytest.raises(PolicyError):
            await reg.invoke(
                "run_stage", Phase.OBSERVE, pipeline_id=pipeline.id, stage_id="orders_clean"
            )
        out = await reg.invoke(
            "run_stage", Phase.VALIDATE, approval=ApprovalStatus.APPROVED,
            pipeline_id=pipeline.id, stage_id="orders_clean",
        )
        assert out["namespace"] == "shadow"


class TestRegistrySafety:
    def test_unknown_tool_rejected_by_policy(self, setup):
        _, _, reg = setup
        with pytest.raises(PolicyError):
            reg.invoke("drop_table", Phase.OBSERVE)

    @pytest.mark.asyncio
    async def test_unregistered_tool_raises(self, setup):
        _, _, reg = setup
        with pytest.raises(KeyError):
            await reg.invoke("profile_dataset", Phase.OBSERVE, pipeline_id="nope")

    def test_all_tools_classified(self, setup):
        _, _, reg = setup
        names = reg.names()
        assert len(names) >= 20
        assert "rollback_pipeline" in names and "run_canary" in names
