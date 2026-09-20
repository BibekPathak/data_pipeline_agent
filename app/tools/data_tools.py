"""Data tools: profiling, sampling, querying, quality checks, distribution comparison."""

from __future__ import annotations

import polars as pl

from app.agent.policies import ActionClass
from app.data.anomaly import detect_anomalies
from app.data.profiler import profile_dataset, value_counts
from app.data.quality import run_quality_checks
from app.tools.base import tool
from app.tools.context import ToolContext


class DataTools:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def tools(self) -> list:
        return [
            self.profile_dataset,
            self.query_dataset,
            self.sample_rows,
            self.run_quality_checks_tool,
            self.compare_distributions,
            self.get_metric_history,
        ]

    @tool("profile_dataset", ActionClass.READ_ONLY, "Profile a dataset (rows, per-column stats).")
    async def profile_dataset(self, pipeline_id: str) -> dict:
        df = self.ctx.observed[pipeline_id]
        return profile_dataset(df)

    @tool("query_dataset", ActionClass.READ_ONLY, "Run a read-only projection/filter over observed data.")
    async def query_dataset(self, pipeline_id: str, columns: list[str] | None = None,
                            limit: int = 100) -> dict:
        df = self.ctx.observed[pipeline_id]
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df.select(keep) if keep else df
        return df.head(limit).to_dict(as_series=False)

    @tool("sample_rows", ActionClass.READ_ONLY, "Sample representative rows from a column.")
    async def sample_rows(self, pipeline_id: str, column: str = "", limit: int = 5) -> dict:
        df = self.ctx.observed[pipeline_id]
        if column and column in df.columns:
            sample = df.select(column).slice(0, limit)
        else:
            sample = df.slice(0, limit)
        return {"rows": sample.to_dict(as_series=False), "column": column}

    @tool("run_quality_checks", ActionClass.READ_ONLY, "Run data-quality checks and return anomalies.")
    async def run_quality_checks_tool(self, pipeline_id: str, referenced: dict | None = None) -> dict:
        df = self.ctx.observed[pipeline_id]
        refs = None
        if referenced:
            refs = {k: self.ctx.observed[k] for k in referenced if k in self.ctx.observed}
        report, anomalies = run_quality_checks(
            df, pipeline_id=pipeline_id, referenced=refs
        )
        return {
            "checks": [c.model_dump() for c in report.checks],
            "passed": report.passed,
            "anomalies": [a.model_dump() for a in anomalies],
        }

    @tool("compare_distributions", ActionClass.READ_ONLY,
          "Compare column distribution to a baseline.")
    async def compare_distributions(self, pipeline_id: str, column: str) -> dict:
        df = self.ctx.observed[pipeline_id]
        history = await self.ctx.store.metadata.get_metric_history(column or pipeline_id)
        anomalies = detect_anomalies(df, history)
        return {
            "column": column,
            "value_counts": value_counts(df, column) if column in df.columns else [],
            "anomalies": [a.model_dump() for a in anomalies],
            "history": history,
        }

    @tool("get_metric_history", ActionClass.READ_ONLY,
          "Return historical metric snapshots for a table.")
    async def get_metric_history(self, table: str) -> dict:
        history = await self.ctx.store.metadata.get_metric_history(table)
        return {"table": table, "history": history}
