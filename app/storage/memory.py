"""In-memory storage backend.

Used by unit/integration tests and fast deterministic demo runs. Implements the
same :class:`MetadataStore` / :class:`Warehouse` protocols as SQLite without any
filesystem or process dependency.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from app.models import Pipeline, PipelineTriageState
from app.models.quality import QualityReport
from app.models.schema import DriftEvent, SchemaDefinition
from app.storage.base import MetadataStore, Warehouse


class MemoryMetadataStore(MetadataStore):
    def __init__(self) -> None:
        self._states: dict[str, PipelineTriageState] = {}
        self._pipelines: dict[str, Pipeline] = {}
        self._schemas: dict[str, SchemaDefinition] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._metrics: list[dict[str, Any]] = []
        self._lineage_edges: list[dict[str, Any]] = []
        self._rollbacks: list[dict[str, Any]] = []

    async def save_state(self, state: PipelineTriageState) -> None:
        self._states[state.run_id] = state

    async def load_state(self, run_id: str) -> PipelineTriageState | None:
        return self._states.get(run_id)

    async def save_pipeline(self, pipeline: Pipeline) -> None:
        self._pipelines[pipeline.id] = pipeline

    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None:
        return self._pipelines.get(pipeline_id)

    async def list_pipelines(self) -> list[Pipeline]:
        return list(self._pipelines.values())

    async def save_schema(self, schema: SchemaDefinition) -> None:
        self._schemas[f"{schema.table}:{schema.version}"] = schema

    async def get_schema(self, table: str, version: int) -> SchemaDefinition | None:
        return self._schemas.get(f"{table}:{version}")

    async def list_schema_versions(self, table: str) -> list[int]:
        versions: list[int] = []
        prefix = f"{table}:"
        for key in self._schemas:
            if key.startswith(prefix):
                versions.append(int(key[len(prefix) :]))
        return sorted(versions)

    async def record_run(
        self,
        run_id: str,
        pipeline_id: str,
        status: str,
        detected_issues: list[DriftEvent] | None = None,
        quality_report: QualityReport | None = None,
    ) -> None:
        self._runs[run_id] = {
            "run_id": run_id,
            "pipeline_id": pipeline_id,
            "status": status,
            "detected_issues": detected_issues or [],
            "quality_report": quality_report,
        }

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    async def record_metrics(
        self, table: str, timestamp: str, metrics: dict[str, float]
    ) -> None:
        self._metrics.append({"table": table, "timestamp": timestamp, **metrics})

    async def get_metric_history(self, table: str) -> list[dict[str, Any]]:
        return [m for m in self._metrics if m["table"] == table]

    async def save_lineage_edge(
        self, source: str, destination: str, columns: list[str]
    ) -> None:
        self._lineage_edges.append(
            {"source": source, "destination": destination, "columns": columns}
        )

    async def get_downstream(self, node: str) -> list[dict[str, Any]]:
        return [e for e in self._lineage_edges if e["source"] == node]

    async def record_rollback(self, record: dict[str, Any]) -> None:
        self._rollbacks.append(record)

    async def get_rollback_records(
        self, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        if run_id is None:
            return list(self._rollbacks)
        return [r for r in self._rollbacks if r.get("run_id") == run_id]


class MemoryWarehouse(Warehouse):
    def __init__(self) -> None:
        self._tables: dict[str, dict[str, pl.DataFrame]] = {}

    async def write_table(
        self, table: str, version: str, df: pl.DataFrame
    ) -> None:
        self._tables.setdefault(table, {})[version] = df

    async def read_table(self, table: str, version: str) -> pl.DataFrame | None:
        return self._tables.get(table, {}).get(version)

    async def list_versions(self, table: str) -> list[str]:
        return list(self._tables.get(table, {}))

    async def drop_version(self, table: str, version: str) -> None:
        self._tables.get(table, {}).pop(version, None)

    async def drop_table(self, table: str) -> None:
        self._tables.pop(table, None)
