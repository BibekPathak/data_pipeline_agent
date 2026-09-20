"""End-to-end orchestrator tests: bounded state machine behavior."""

from __future__ import annotations

import polars as pl
import pytest

from app.agent.llm import DeterministicLLM
from app.agent.orchestrator import Orchestrator, TriageConfig
from app.data.lineage import LineageEdge, LineageGraph
from app.models import (
    ColumnDefinition,
    DataType,
    FixOperation,
    FixOperationType,
    Pipeline,
    PipelineStage,
    SchemaDefinition,
)
from app.models.state import ApprovalStatus, DeploymentStatus, Phase
from app.storage import BackendStore, MemoryMetadataStore, MemoryWarehouse
from app.tools.context import ToolContext


def _sch(v: int, types) -> SchemaDefinition:
    return SchemaDefinition(
        table="orders", version=v,
        columns=[ColumnDefinition(name=n, type=t) for n, t in types],
    )


def _pipeline() -> Pipeline:
    exp = _sch(3, [("order_id", DataType.INTEGER), ("amount", DataType.DOUBLE), ("currency", DataType.VARCHAR)])
    return Pipeline(
        id="orders", name="Orders", version="1.0", source="x",
        stages=[
            PipelineStage(id="orders_clean", name="clean.orders",
                          input_schema=exp, output_schema=exp,
                          transformation="clean_orders", dependencies=[]),
            PipelineStage(id="daily_revenue", name="daily_revenue",
                          input_schema=exp,
                          output_schema=_sch(1, [("order_id", DataType.INTEGER),
                                                  ("currency", DataType.VARCHAR),
                                                  ("revenue", DataType.DOUBLE)]),
                          transformation="aggregate_daily_revenue",
                          dependencies=["orders_clean"]),
        ],
    )


def _ctx() -> ToolContext:
    store = BackendStore(MemoryMetadataStore(), MemoryWarehouse())
    lineage = LineageGraph([
        LineageEdge("orders", "clean.orders", ["order_id", "amount", "currency"]),
        LineageEdge("clean.orders", "daily_revenue", ["amount"]),
    ])
    return ToolContext(store=store, pipelines={"orders": _pipeline()}, lineage=lineage)


def _drifted_df(n: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "order_id": list(range(n)),
            "amount": [str(10 + i * 0.5) for i in range(n)],
            "currency": ["USD" if i % 3 else "EUR" for i in range(n)],
        }
    )


class TestTypeDriftPromotion:
    @pytest.mark.asyncio
    async def test_drift_reaches_success(self):
        ctx = _ctx()
        orch = Orchestrator(ctx=ctx, llm=DeterministicLLM(), config=TriageConfig(approval_mode="auto"))
        state = await orch.run("orders", _drifted_df())

        assert state.phase == Phase.SUCCESS
        assert state.deployment_status == DeploymentStatus.ACTIVE
        assert any(e.column == "amount" for e in state.detected_issues)
        assert state.validation_report is not None and state.validation_report.passed
        assert state.approval_status == ApprovalStatus.AUTO_APPROVED
        # State is persisted/resumable.
        assert await ctx.store.metadata.load_state(state.run_id) is not None

    @pytest.mark.asyncio
    async def test_state_phase_transitions_recorded(self):
        ctx = _ctx()
        orch = Orchestrator(ctx=ctx, llm=DeterministicLLM(), config=TriageConfig())
        state = await orch.run("orders", _drifted_df())
        assert state.phase == Phase.SUCCESS
        # Proposed fix cast the drifted type back to DOUBLE.
        assert state.proposed_fix is not None
        assert state.proposed_fix.operations[0].operation == FixOperationType.CAST_TYPE


class TestGuardrails:
    @pytest.mark.asyncio
    async def test_resource_budget_safe_stop(self):
        ctx = _ctx()
        orch = Orchestrator(
            ctx=ctx,
            llm=DeterministicLLM(),
            config=TriageConfig(max_budget_rows=10),
        )
        state = await orch.run("orders", _drifted_df(n=100))
        assert state.phase == Phase.SAFE_STOP
        assert "budget" in (state.error or "")

    @pytest.mark.asyncio
    async def test_canary_failure_triggers_rollback(self, monkeypatch):
        from app.agent.policies import ActionClass
        from app.tools.base import tool
        from app.tools.deployment_tools import DeploymentTools

        @tool("run_canary", ActionClass.PRODUCTION_WRITE, "force-fail canary")
        async def broken_canary(self, pipeline_id, operations=None):
            return {"status": "CANARY_FAILED", "passed": False, "failure_rate": 1.0}

        monkeypatch.setattr(DeploymentTools, "run_canary", broken_canary)

        ctx = _ctx()
        orch = Orchestrator(ctx=ctx, llm=DeterministicLLM(), config=TriageConfig())
        state = await orch.run("orders", _drifted_df())
        assert state.deployment_status == DeploymentStatus.ROLLED_BACK
        assert state.rollback_status is not None


class FakeLLM:
    def __init__(self, proposal):
        self._proposal = proposal

    def name(self):
        return "fake"

    async def build_proposal(self, **kwargs):
        return self._proposal


class TestUnsafeFixRejected:
    @pytest.mark.asyncio
    async def test_destructive_proposal_rejected_no_mutation(self):
        from app.models import FixProposal, RiskLevel

        ctx = _ctx()
        # Destructive: cast safely, then drop rows -> classification is HIGH.
        bad = FakeLLM(
            FixProposal(
                root_cause="discard data",
                operations=[
                    FixOperation(operation=FixOperationType.CAST_TYPE, column="amount", to_type="DOUBLE"),
                    FixOperation(operation=FixOperationType.DROP_INVALID_ROWS, column="amount"),
                ],
                risk=RiskLevel.HIGH,
            )
        )
        orch = Orchestrator(ctx=ctx, llm=bad, config=TriageConfig())
        state = await orch.run("orders", _drifted_df())
        assert state.phase == Phase.SAFE_STOP
        assert state.deployment_status == DeploymentStatus.NOT_DEPLOYED
        assert "rejected" in (state.final_result or "")
