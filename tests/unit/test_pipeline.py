"""Unit tests for the pipeline package: stages, fixes, runner, registry."""

from __future__ import annotations

import json

import polars as pl
import pytest

from app.models import (
    ColumnDefinition,
    DataType,
    FixOperation,
    FixOperationType,
    OnError,
    Pipeline,
    PipelineStage,
    SchemaDefinition,
)
from app.pipeline.registry import PipelineRegistry
from app.pipeline.runner import run_pipeline
from app.pipeline.stages import (
    TransformationError,
    apply_fix_operations,
    list_stage_transforms,
    op_cast_type,
    op_parse_timestamp,
)
from app.storage import MemoryWarehouse


def _orders_schema(v: int = 3, amount: DataType = DataType.DOUBLE) -> SchemaDefinition:
    return SchemaDefinition(
        table="orders",
        version=v,
        columns=[
            ColumnDefinition(name="order_id", type=DataType.INTEGER),
            ColumnDefinition(name="amount", type=amount),
            ColumnDefinition(name="currency", type=DataType.VARCHAR),
        ],
    )


class TestSchemaInference:
    def test_cast_type_fix(self):
        op = FixOperation(
            operation=FixOperationType.CAST_TYPE,
            column="amount",
            to_type="DOUBLE",
            on_error=OnError.REJECT,
        )
        df = pl.DataFrame({"order_id": [1, 2], "amount": ["10.5", "20.0"]})
        out = apply_fix_operations(df, [op])
        assert out["amount"].dtype.is_float()
        assert out["amount"].to_list() == [10.5, 20.0]

    def test_rename_and_timestamp(self):
        df = pl.DataFrame(
            {
                "user_id": [1, 2],
                "ts": ["2024-01-01 10:00:00", "2024-01-02 10:00:00"],
            }
        )
        ops = [
            FixOperation(
                operation=FixOperationType.RENAME_COLUMN,
                column="user_id",
                target="customer_id",
            ),
            FixOperation(operation=FixOperationType.PARSE_TIMESTAMP, column="ts"),
        ]
        out = apply_fix_operations(df, ops)
        assert "customer_id" in out.columns
        assert "user_id" not in out.columns
        assert out["ts"].dtype == pl.Datetime

    def test_fill_null_and_deduplicate(self):
        df = pl.DataFrame({"a": [1, 1, 2, None, 3], "b": [1, 1, 2, None, 3]})
        ops = [
            FixOperation(
                operation=FixOperationType.FILL_NULL, column="b", value=0
            ),
            FixOperation(operation=FixOperationType.DEDUPLICATE),
        ]
        out = apply_fix_operations(df, ops)
        assert out["b"].null_count() == 0
        assert out.height == df.unique().height

    def test_normalize_and_column_mapping(self):
        df = pl.DataFrame(
            {"code": ["  USD  ", " EUR ", ""], "status": ["A", "B", "C"]}
        )
        ops = [
            FixOperation(
                operation=FixOperationType.NORMALIZE_STRING,
                column="code",
                target="strip",
            ),
            FixOperation(
                operation=FixOperationType.COLUMN_MAPPING,
                column="status",
                value={"A": "active", "B": "blocked", "C": "closed"},
            ),
        ]
        out = apply_fix_operations(df, ops)
        assert out["code"].to_list() == ["USD", "EUR", ""]
        assert out["status"].to_list() == ["active", "blocked", "closed"]

    def test_default_value_replaces_empty(self):
        df = pl.DataFrame({"tier": [None, "", "gold"]})
        out = apply_fix_operations(
            df,
            [FixOperation(operation=FixOperationType.DEFAULT_VALUE, column="tier", value="standard")],
        )
        assert out["tier"].null_count() == 0
        assert out["tier"].to_list() == ["standard", "standard", "gold"]


class TestStageOps:
    def test_cast_drop(self):
        df = pl.DataFrame({"amount": ["1.0", "2.0", "bad", "3.0"]})
        out = op_cast_type(df, "amount", "DOUBLE", on_error=OnError.DROP)
        assert out.height == 3
        assert out["amount"].dtype.is_float()

    def test_cast_null(self):
        df = pl.DataFrame({"amount": ["1.0", "bad", "3.0"]})
        out = op_cast_type(df, "amount", "DOUBLE", on_error=OnError.NULL)
        assert out.height == 3
        assert out["amount"].null_count() == 1

    def test_cast_reject_raises(self):
        df = pl.DataFrame({"amount": ["1.0", "bad"]})
        with pytest.raises(TransformationError):
            op_cast_type(df, "amount", "DOUBLE", on_error=OnError.REJECT)

    def test_parse_timestamp(self):
        df = pl.DataFrame({"ts": ["2024-01-01", "2024-01-02"]})
        out = op_parse_timestamp(df, "ts")
        assert out["ts"].dtype in (pl.Datetime, pl.Date)

    def test_known_stage_transforms(self):
        names = list_stage_transforms()
        assert "clean_orders" in names
        assert "aggregate_daily_revenue" in names


class TestRunner:
    def _pipeline(self) -> Pipeline:
        return Pipeline(
            id="orders",
            name="Orders",
            version="1.0",
            source="fixtures:orders.csv",
            stages=[
                PipelineStage(
                    id="orders_clean",
                    name="clean.orders",
                    input_schema=_orders_schema(3),
                    output_schema=_orders_schema(3),
                    transformation="clean_orders",
                    dependencies=[],
                ),
                PipelineStage(
                    id="daily_revenue",
                    name="daily_revenue",
                    input_schema=_orders_schema(3),
                    output_schema=SchemaDefinition(
                        table="daily_revenue",
                        version=1,
                        columns=[
                            ColumnDefinition(name="order_id", type=DataType.INTEGER),
                            ColumnDefinition(name="currency", type=DataType.VARCHAR),
                            ColumnDefinition(name="revenue", type=DataType.DOUBLE),
                        ],
                    ),
                    transformation="aggregate_daily_revenue",
                    dependencies=["orders_clean"],
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_run_success_and_warehouse_write(self):
        wh = MemoryWarehouse()
        df = pl.DataFrame(
            {
                "order_id": [1, 1, 2, 3],
                "amount": [10.0, 20.0, 30.0, 40.0],
                "currency": ["USD", "USD", "EUR", "GBP"],
            }
        )
        res = await run_pipeline(self._pipeline(), df, wh, namespace="shadow")
        assert res.ok is True
        assert [s.ok for s in res.stages] == [True, True]
        assert (await wh.list_versions("daily_revenue")) == ["shadow:v1.0"]

    @pytest.mark.asyncio
    async def test_run_persist_false_no_write(self):
        wh = MemoryWarehouse()
        df = pl.DataFrame(
            {"order_id": [1], "amount": [1.0], "currency": ["USD"]}
        )
        res = await run_pipeline(
            self._pipeline(), df, wh, namespace="shadow", persist=False
        )
        assert res.ok is True
        assert (await wh.list_versions("daily_revenue")) == []

    @pytest.mark.asyncio
    async def test_unknown_transform_errors(self):
        p = Pipeline(
            id="bad",
            name="Bad",
            version="1.0",
            source="x",
            stages=[
                PipelineStage(
                    id="s1",
                    name="x",
                    input_schema=_orders_schema(),
                    output_schema=_orders_schema(),
                    transformation="does_not_exist",
                )
            ],
        )
        res = await run_pipeline(p, pl.DataFrame({}), None)
        assert res.ok is False
        assert res.failed_stages()[0].stage_id == "s1"


class TestRegistry:
    def test_register_and_override(self, tmp_path):
        reg = PipelineRegistry(tmp_path)
        p = Pipeline(id="a", name="A", version="1.0", source="x", stages=[])
        reg.register(p)
        assert reg.get("a") == p
        assert len(reg) == 1

    def test_load_from_disk(self, tmp_path):
        p = Pipeline(id="disk", name="Disk", version="1.0", source="x", stages=[])
        (tmp_path / "disk.json").write_text(json.dumps(p.model_dump()))
        reg = PipelineRegistry(tmp_path)
        assert reg.get("disk").id == "disk"
