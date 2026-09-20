"""Storage protocols.

The system separates two concerns, each behind a protocol so the concrete
backends (SQLite, in-memory, and later DuckDB/Postgres) are swappable:

- ``MetadataStore``: persists agent triage state, pipeline runs, schema
  registry versions, quality/anomaly metric history, lineage edges and
  rollback records. This is what makes the agent *resumable*.
- ``Warehouse``: stores the actual datasets (Polars DataFrames) keyed by
  ``(table, version)``. Versioned writes give canary deployment and rollback
  a deterministic, idempotent primitive.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import polars as pl

from app.models import Pipeline, PipelineTriageState
from app.models.quality import QualityReport
from app.models.schema import DriftEvent, SchemaDefinition


@runtime_checkable
class MetadataStore(Protocol):
    """Persistence for operational metadata. Async methods."""

    # --- Agent state (resumability) ---
    async def save_state(self, state: PipelineTriageState) -> None: ...
    async def load_state(self, run_id: str) -> PipelineTriageState | None: ...

    # --- Pipeline registry ---
    async def save_pipeline(self, pipeline: Pipeline) -> None: ...
    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None: ...
    async def list_pipelines(self) -> list[Pipeline]: ...

    # --- Schema registry (historical versions) ---
    async def save_schema(self, schema: SchemaDefinition) -> None: ...
    async def get_schema(self, table: str, version: int) -> SchemaDefinition | None: ...
    async def list_schema_versions(self, table: str) -> list[int]: ...

    # --- Pipeline runs ---
    async def record_run(
        self,
        run_id: str,
        pipeline_id: str,
        status: str,
        detected_issues: list[DriftEvent] | None = None,
        quality_report: QualityReport | None = None,
    ) -> None: ...
    async def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    # --- Metric history (anomaly detection evidence) ---
    async def record_metrics(
        self, table: str, timestamp: str, metrics: dict[str, float]
    ) -> None: ...
    async def get_metric_history(self, table: str) -> list[dict[str, Any]]: ...

    # --- Lineage ---
    async def save_lineage_edge(
        self, source: str, destination: str, columns: list[str]
    ) -> None: ...
    async def get_downstream(self, node: str) -> list[dict[str, Any]]: ...

    # --- Rollback records ---
    async def record_rollback(self, record: dict[str, Any]) -> None: ...
    async def get_rollback_records(self, run_id: str | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class Warehouse(Protocol):
    """Versioned dataset store. DataFrames travel in, keyed by table+version."""

    async def write_table(
        self, table: str, version: str, df: pl.DataFrame
    ) -> None: ...
    async def read_table(self, table: str, version: str) -> pl.DataFrame | None: ...
    async def list_versions(self, table: str) -> list[str]: ...
    async def drop_version(self, table: str, version: str) -> None: ...
    async def drop_table(self, table: str) -> None: ...


class BackendStore(ABC):
    """Combined convenience wrapper around metadata + warehouse backends."""

    def __init__(self, metadata: MetadataStore, warehouse: Warehouse) -> None:
        self.metadata = metadata
        self.warehouse = warehouse
