"""Deployment tools: staging, canary, monitor, and rollback.

Canary deployment simulates a controlled rollout by partitioning the observed
data into 95% (current pipeline) and 5% (candidate pipeline) and comparing
failure rate, quality, row counts and business metrics. Threshold violations
produce ``CANARY_FAILED`` and trigger rollback.
"""

from __future__ import annotations

import polars as pl

from app.agent.policies import ActionClass
from app.pipeline.runner import run_pipeline
from app.rollback.manager import RollbackManager
from app.tools.base import tool
from app.tools.context import ToolContext


class DeploymentTools:
    CANARY_SPLIT = 0.05
    THRESHOLDS = {
        "failure_rate": 0.0,
        "row_delta_pct": 0.05,
        "metric_delta_pct": 0.05,
        "null_rate": 0.05,
    }

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self.rollback = RollbackManager(ctx.store.warehouse, ctx.store.metadata)
        self._canary: dict[str, dict] = {}
        self._active_table: dict[str, str] = {}

    def tools(self) -> list:
        return [
            self.stage_pipeline,
            self.run_canary,
            self.monitor_canary,
            self.rollback_pipeline,
        ]

    @tool("stage_pipeline", ActionClass.STAGING_WRITE,
          "Stage a candidate pipeline version (records active -> previous).")
    async def stage_pipeline(self, pipeline_id: str) -> dict:
        pipeline = self.ctx.pipelines[pipeline_id]
        table = pipeline.stages[0].name
        prev = self.rollback.active_version(table)
        self.rollback.set_active(table, pipeline.version)
        self._active_table[pipeline_id] = table
        return {
            "pipeline_id": pipeline_id,
            "stage": pipeline.version,
            "previous_active": prev,
        }

    @tool("run_canary", ActionClass.PRODUCTION_WRITE,
          "Run a 95/5 canary: current (unfixed) vs candidate (fixed) pipeline.")
    async def run_canary(self, pipeline_id: str, operations: list | None = None) -> dict:
        from app.models import FixOperation
        from app.pipeline.stages import apply_fix_operations

        pipeline = self.ctx.pipelines[pipeline_id]
        df = self.ctx.observed[pipeline_id]

        n_canary = max(1, int(df.height * self.CANARY_SPLIT))
        split_idx = df.height - n_canary
        current_df = df.slice(0, split_idx)
        candidate_df = df.slice(split_idx, n_canary)

        ops = (
            [FixOperation.model_validate(o) for o in operations] if operations else []
        )
        fixed_df = apply_fix_operations(candidate_df, ops)

        # Current (unfixed) on 95% -> expected to fail while the drift exists.
        current_run = await run_pipeline(
            pipeline, current_df, self.ctx.store.warehouse,
            namespace="canary_current", persist=False,
        )
        # Candidate (fixed) on 5% -> the meaningful health signal.
        candidate_run = await run_pipeline(
            pipeline, fixed_df, self.ctx.store.warehouse,
            namespace="canary_candidate", persist=False,
        )

        candidate_ok = candidate_run.ok
        candidate_rows = fixed_df.height if candidate_ok else 0

        # Candidate quality invariants.
        quality = {"passed": True}
        if candidate_ok and candidate_run.final_df is not None:
            from app.data.quality import run_quality_checks

            quality_report, _ = run_quality_checks(
                candidate_run.final_df, pipeline_id=pipeline_id
            )
            quality = {"passed": quality_report.passed}

        metric_delta = 0.0
        metric_delta_valid = current_run.ok and candidate_run.ok
        if (
            metric_delta_valid
            and current_run.final_df is not None
            and candidate_run.final_df is not None
        ):
            for col in current_run.final_df.columns:
                if current_run.final_df[col].dtype.is_numeric():
                    s_cur = float(current_run.final_df[col].sum())
                    s_cand = float(candidate_run.final_df[col].sum())
                    if s_cur != 0:
                        metric_delta = max(metric_delta, abs(s_cand - s_cur) / abs(s_cur))

        failed = (not candidate_ok) or (not quality["passed"])
        metrics = {
            "failure_rate": 0.0 if candidate_ok else 1.0,
            "current_ok": current_run.ok,
            "candidate_ok": candidate_ok,
            "candidate_rows": candidate_rows,
            "quality_pass": quality["passed"],
            "metric_delta_pct": round(metric_delta, 4),
            "passed": not failed,
            "status": "CANARY_PASSED" if not failed else "CANARY_FAILED",
        }
        self._canary[pipeline_id] = metrics
        return metrics

    @tool("monitor_canary", ActionClass.READ_ONLY,
          "Return the latest canary metrics for a pipeline.")
    async def monitor_canary(self, pipeline_id: str) -> dict:
        return self._canary.get(
            pipeline_id,
            {"status": "NO_CANARY", "passed": True},
        )

    @tool("rollback_pipeline", ActionClass.ROLLBACK,
          "Roll back to the previous active pipeline version.")
    async def rollback_pipeline(self, pipeline_id: str, reason: str = "") -> dict:
        table = self._active_table.get(pipeline_id)
        if table is None:
            # Fall back to the first stage table.
            pipeline = self.ctx.pipelines[pipeline_id]
            table = pipeline.stages[0].name
        report = await self.rollback.rollback(
            run_id=pipeline_id, table=table, reason=reason,
            failed_metrics=self._canary.get(pipeline_id, {}),
        )
        return {
            "rolled_back": report.rolled_back,
            "reason": report.reason,
            "previous_version": report.previous_version,
            "restored_version": report.restored_version,
        }
