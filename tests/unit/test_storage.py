"""Unit tests for the storage layer: both backends (in-memory + SQLite)."""

from __future__ import annotations

import polars as pl
import pytest

from app.models import (
    ColumnDefinition,
    DataType,
    DriftEvent,
    DriftEventType,
    Phase,
    Pipeline,
    PipelineStage,
    PipelineTriageState,
    SchemaDefinition,
    Severity,
)
from app.storage import (
    MemoryMetadataStore,
    MemoryWarehouse,
    SQLiteMetadataStore,
    SQLiteWarehouse,
    create_store,
)


def _orders_schema(version: int = 3) -> SchemaDefinition:
    return SchemaDefinition(
        table="orders",
        version=version,
        columns=[
            ColumnDefinition(name="order_id", type=DataType.INTEGER),
            ColumnDefinition(name="amount", type=DataType.DOUBLE),
            ColumnDefinition(name="currency", type=DataType.VARCHAR),
        ],
    )


def _orders_pipeline() -> Pipeline:
    stage = PipelineStage(
        id="orders_clean",
        name="Orders Clean",
        input_schema=_orders_schema(),
        output_schema=_orders_schema(version=4),
        transformation="clean_orders",
    )
    return Pipeline(
        id="orders",
        name="Orders Pipeline",
        version="1.0",
        source="fixtures:orders.csv",
        stages=[stage],
    )


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "amount": [49.99, 12.0, 99.0, 4.5],
            "currency": ["USD", "USD", "EUR", "GBP"],
        }
    )


METADATA_BACKENDS = [
    pytest.param(MemoryMetadataStore, id="memory"),
    pytest.param(
        lambda: SQLiteMetadataStore(":memory:", create=True), id="sqlite"
    ),
]

WAREHOUSE_BACKENDS = [
    pytest.param(MemoryWarehouse, id="memory"),
    pytest.param(
        lambda: SQLiteWarehouse(":memory:", create=True), id="sqlite"
    ),
]


class TestMetadataStore:
    @pytest.mark.parametrize("factory", METADATA_BACKENDS)
    @pytest.mark.asyncio
    async def test_state_roundtrip(self, factory):
        store = factory()
        st = PipelineTriageState(run_id="run1", pipeline_id="orders")
        st.phase = Phase.DIAGNOSE
        await store.save_state(st)
        loaded = await store.load_state("run1")
        assert loaded is not None
        assert loaded.run_id == "run1"
        assert loaded.phase == Phase.DIAGNOSE

        assert await store.load_state("nope") is None

    @pytest.mark.parametrize("factory", METADATA_BACKENDS)
    @pytest.mark.asyncio
    async def test_pipeline_crud(self, factory):
        store = factory()
        p = _orders_pipeline()
        await store.save_pipeline(p)
        assert await store.get_pipeline("orders") == p
        assert [x.id for x in await store.list_pipelines()] == ["orders"]

    @pytest.mark.parametrize("factory", METADATA_BACKENDS)
    @pytest.mark.asyncio
    async def test_schema_registry_versions(self, factory):
        store = factory()
        await store.save_schema(_orders_schema(version=1))
        await store.save_schema(_orders_schema(version=2))
        await store.save_schema(_orders_schema(version=3))
        assert await store.list_schema_versions("orders") == [1, 2, 3]
        got = await store.get_schema("orders", 2)
        assert got is not None and got.version == 2

    @pytest.mark.parametrize("factory", METADATA_BACKENDS)
    @pytest.mark.asyncio
    async def test_run_and_metrics_and_lineage(self, factory):
        store = factory()
        drift = DriftEvent(
            type=DriftEventType.TYPE_CHANGED,
            column="amount",
            expected="DOUBLE",
            observed="VARCHAR",
            severity=Severity.HIGH,
        )
        await store.record_run("r1", "orders", "failed", detected_issues=[drift])
        run = await store.get_run("r1")
        assert run["pipeline_id"] == "orders"

        await store.record_metrics("orders", "t1", {"row_count": 100, "null_rate": 0.02})
        await store.record_metrics("orders", "t2", {"row_count": 90, "null_rate": 0.03})
        hist = await store.get_metric_history("orders")
        assert len(hist) == 2
        assert hist[0]["row_count"] == 100

        await store.save_lineage_edge("raw.orders", "clean.orders", ["amount"])
        downstream = await store.get_downstream("raw.orders")
        assert downstream[0]["destination"] == "clean.orders"
        assert downstream[0]["columns"] == ["amount"]

    @pytest.mark.parametrize("factory", METADATA_BACKENDS)
    @pytest.mark.asyncio
    async def test_rollback_records(self, factory):
        store = factory()
        await store.record_rollback(
            {"run_id": "r1", "reason": "canary failed", "prev": "v3", "restored": "v3"}
        )
        recs = await store.get_rollback_records(run_id="r1")
        assert len(recs) == 1
        assert recs[0]["restored"] == "v3"


class TestWarehouse:
    @pytest.mark.parametrize("factory", WAREHOUSE_BACKENDS)
    @pytest.mark.asyncio
    async def test_versioned_roundtrip(self, factory):
        wh = factory()
        df = _sample_df()
        await wh.write_table("orders", "v3", df)
        await wh.write_table("orders", "v4", df.with_columns(pl.lit(1).alias("x")))
        restored = await wh.read_table("orders", "v3")
        assert restored is not None
        assert restored.height == 4
        assert "amount" in restored.columns
        assert await wh.list_versions("orders") == ["v3", "v4"]
        assert await wh.read_table("orders", "nope") is None

    @pytest.mark.parametrize("factory", WAREHOUSE_BACKENDS)
    @pytest.mark.asyncio
    async def test_drop_version_and_table(self, factory):
        wh = factory()
        await wh.write_table("t", "v1", _sample_df())
        await wh.write_table("t", "v2", _sample_df())
        await wh.drop_version("t", "v1")
        assert await wh.list_versions("t") == ["v2"]
        await wh.drop_table("t")
        assert await wh.list_versions("t") == []


class TestBackendFactory:
    @pytest.mark.asyncio
    async def test_factory_memory(self):
        store = create_store("memory")
        st = PipelineTriageState(run_id="r", pipeline_id="orders")
        await store.metadata.save_state(st)
        assert (await store.metadata.load_state("r")) is not None
        await store.warehouse.write_table("o", "v1", _sample_df())
        assert (await store.warehouse.read_table("o", "v1")).height == 4

    @pytest.mark.asyncio
    async def test_factory_sqlite(self, tmp_path):
        db = tmp_path / "p.db"
        store = create_store("sqlite", str(db))
        await store.metadata.save_state(
            PipelineTriageState(run_id="r", pipeline_id="orders")
        )
        assert (await store.metadata.load_state("r")) is not None
        await store.warehouse.write_table("o", "v1", _sample_df())
        assert (await store.warehouse.read_table("o", "v1")).height == 4

    def test_factory_unknown(self):
        with pytest.raises(ValueError):
            create_store("postgres")

    @pytest.mark.asyncio
    async def test_store_from_settings(self, tmp_path):
        from app.config import Settings, StorageBackend
        from app.storage import store_from_settings

        s = Settings(
            _env_file=None,
            storage_backend=StorageBackend.MEMORY,
            db_path=tmp_path / "ignored.db",
        )
        store = store_from_settings(s)
        await store.metadata.save_state(
            PipelineTriageState(run_id="x", pipeline_id="orders")
        )
        assert (await store.metadata.load_state("x")) is not None
