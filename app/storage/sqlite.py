"""SQLite storage backend.

Implements both :class:`MetadataStore` and :class:`Warehouse` on top of a single
SQLite database file (SQLAlchemy ORM for metadata, Parquet blobs for versioned
warehouse datasets). Warehouse rows are stored as lossless Parquet blobs so
Polars dtypes survive the round trip, giving deterministic snapshot/restore
behaviour for canary and rollback.

DuckDB/Postgres adapters can replace this class later without touching callers.
"""

from __future__ import annotations

import io
import json
from typing import Any

import polars as pl
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    LargeBinary,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.models import Pipeline, PipelineTriageState
from app.models.quality import QualityReport
from app.models.schema import DriftEvent, SchemaDefinition
from app.storage.base import MetadataStore, Warehouse


class Base(DeclarativeBase):
    pass


class AgentStateRow(Base):
    __tablename__ = "agent_state"
    run_id = Column(String, primary_key=True)
    state_json = Column(Text)


class PipelineRow(Base):
    __tablename__ = "pipelines"
    id = Column(String, primary_key=True)
    pipeline_json = Column(Text)


class SchemaRow(Base):
    __tablename__ = "schema_registry"
    table = Column(String, primary_key=True)
    version = Column(Integer, primary_key=True)
    schema_json = Column(Text)


class RunRow(Base):
    __tablename__ = "pipeline_runs"
    run_id = Column(String, primary_key=True)
    pipeline_id = Column(String)
    status = Column(String)
    issues_json = Column(Text, default="[]")
    quality_json = Column(Text, nullable=True)


class MetricRow(Base):
    __tablename__ = "metric_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    table = Column(String, index=True)
    timestamp = Column(String)
    name = Column(String)
    value = Column(Float)


class LineageRow(Base):
    __tablename__ = "lineage"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String, index=True)
    destination = Column(String)
    columns_json = Column(Text, default="[]")


class RollbackRow(Base):
    __tablename__ = "rollbacks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    record_json = Column(Text)


class WarehouseVersionRow(Base):
    __tablename__ = "warehouse_versions"
    table = Column(String, primary_key=True)
    version = Column(String, primary_key=True)
    parquet = Column(LargeBinary)
    row_count = Column(Integer)


class SQLiteMetadataStore(MetadataStore):
    def __init__(self, db_path: str, *, create: bool = True) -> None:
        self.db_path = db_path
        self._engine = create_engine(f"sqlite:///{db_path}", future=True)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )
        if create:
            Base.metadata.create_all(self._engine)

    def _session(self) -> Session:
        return self._session_factory()

    async def save_state(self, state: PipelineTriageState) -> None:
        with self._session() as s:
            row = s.get(AgentStateRow, state.run_id)
            if row is None:
                row = AgentStateRow(run_id=state.run_id)
                s.add(row)
            row.state_json = state.model_dump_json()
            s.commit()

    async def load_state(self, run_id: str) -> PipelineTriageState | None:
        with self._session() as s:
            row = s.get(AgentStateRow, run_id)
        if row is None or not row.state_json:
            return None
        return PipelineTriageState.model_validate_json(row.state_json)

    async def save_pipeline(self, pipeline: Pipeline) -> None:
        with self._session() as s:
            row = s.get(PipelineRow, pipeline.id)
            if row is None:
                row = PipelineRow(id=pipeline.id)
                s.add(row)
            row.pipeline_json = pipeline.model_dump_json()
            s.commit()

    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None:
        with self._session() as s:
            row = s.get(PipelineRow, pipeline_id)
        if row is None or not row.pipeline_json:
            return None
        return Pipeline.model_validate_json(row.pipeline_json)

    async def list_pipelines(self) -> list[Pipeline]:
        with self._session() as s:
            rows = s.execute(select(PipelineRow)).scalars().all()
        return [
            Pipeline.model_validate_json(r.pipeline_json)
            for r in rows
            if r.pipeline_json
        ]

    async def save_schema(self, schema: SchemaDefinition) -> None:
        with self._session() as s:
            row = s.get(SchemaRow, (schema.table, schema.version))
            if row is None:
                row = SchemaRow(table=schema.table, version=schema.version)
                s.add(row)
            row.schema_json = schema.model_dump_json()
            s.commit()

    async def get_schema(self, table: str, version: int) -> SchemaDefinition | None:
        with self._session() as s:
            row = s.get(SchemaRow, (table, version))
        if row is None or not row.schema_json:
            return None
        return SchemaDefinition.model_validate_json(row.schema_json)

    async def list_schema_versions(self, table: str) -> list[int]:
        with self._session() as s:
            rows = s.execute(
                select(SchemaRow.version).where(SchemaRow.table == table)
            ).scalars().all()
        return sorted(rows)

    async def record_run(
        self,
        run_id: str,
        pipeline_id: str,
        status: str,
        detected_issues: list[DriftEvent] | None = None,
        quality_report: QualityReport | None = None,
    ) -> None:
        issues = detected_issues or []
        with self._session() as s:
            row = s.get(RunRow, run_id)
            if row is None:
                row = RunRow(run_id=run_id)
                s.add(row)
            row.pipeline_id = pipeline_id
            row.status = status
            row.issues_json = json.dumps([i.model_dump() for i in issues])
            row.quality_json = (
                quality_report.model_dump_json() if quality_report else None
            )
            s.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._session() as s:
            row = s.get(RunRow, run_id)
        if row is None:
            return None
        return {
            "run_id": row.run_id,
            "pipeline_id": row.pipeline_id,
            "status": row.status,
            "detected_issues": row.issues_json or "[]",
            "quality_report": row.quality_json,
        }

    async def record_metrics(
        self, table: str, timestamp: str, metrics: dict[str, float]
    ) -> None:
        with self._session() as s:
            for name, value in metrics.items():
                s.add(
                    MetricRow(
                        table=table, timestamp=timestamp, name=name, value=value
                    )
                )
            s.commit()

    async def get_metric_history(self, table: str) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.execute(
                select(MetricRow).where(MetricRow.table == table).order_by(MetricRow.id)
            ).scalars().all()
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            out.setdefault(r.name, []).append(
                {"timestamp": r.timestamp, "value": r.value}
            )
        # Return list of per-timestamp snapshots {timestamp: {name: value}}
        by_ts: dict[str, dict[str, Any]] = {}
        for r in rows:
            by_ts.setdefault(r.timestamp, {"timestamp": r.timestamp})[r.name] = r.value
        return list(by_ts.values())

    async def save_lineage_edge(
        self, source: str, destination: str, columns: list[str]
    ) -> None:
        with self._session() as s:
            s.add(
                LineageRow(
                    source=source,
                    destination=destination,
                    columns_json=json.dumps(columns),
                )
            )
            s.commit()

    async def get_downstream(self, node: str) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.execute(
                select(LineageRow).where(LineageRow.source == node)
            ).scalars().all()
        return [
            {
                "source": r.source,
                "destination": r.destination,
                "columns": json.loads(r.columns_json or "[]"),
            }
            for r in rows
        ]

    async def record_rollback(self, record: dict[str, Any]) -> None:
        with self._session() as s:
            s.add(
                RollbackRow(
                    run_id=record.get("run_id", ""),
                    record_json=json.dumps(record),
                )
            )
            s.commit()

    async def get_rollback_records(
        self, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._session() as s:
            if run_id is None:
                rows = s.execute(select(RollbackRow)).scalars().all()
            else:
                rows = s.execute(
                    select(RollbackRow).where(RollbackRow.run_id == run_id)
                ).scalars().all()
        return [json.loads(r.record_json) for r in rows]


class SQLiteWarehouse(Warehouse):
    def __init__(self, db_path: str, *, create: bool = True) -> None:
        self.db_path = db_path
        self._engine = create_engine(f"sqlite:///{db_path}", future=True)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )
        if create:
            Base.metadata.create_all(self._engine)

    async def write_table(self, table: str, version: str, df: pl.DataFrame) -> None:
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        with self._session_factory() as s:
            row = s.get(WarehouseVersionRow, (table, version))
            if row is None:
                row = WarehouseVersionRow(table=table, version=version)
                s.add(row)
            row.parquet = buffer.getvalue()
            row.row_count = df.height
            s.commit()

    async def read_table(self, table: str, version: str) -> pl.DataFrame | None:
        with self._session_factory() as s:
            row = s.get(WarehouseVersionRow, (table, version))
        if row is None or not row.parquet:
            return None
        return pl.read_parquet(io.BytesIO(row.parquet))

    async def list_versions(self, table: str) -> list[str]:
        with self._session_factory() as s:
            rows = s.execute(
                select(WarehouseVersionRow.version).where(
                    WarehouseVersionRow.table == table
                )
            ).scalars().all()
        return list(rows)

    async def drop_version(self, table: str, version: str) -> None:
        with self._session_factory() as s:
            row = s.get(WarehouseVersionRow, (table, version))
            if row is not None:
                s.delete(row)
                s.commit()

    async def drop_table(self, table: str) -> None:
        with self._session_factory() as s:
            rows = s.execute(
                select(WarehouseVersionRow).where(WarehouseVersionRow.table == table)
            ).scalars().all()
            for r in rows:
                s.delete(r)
            s.commit()
