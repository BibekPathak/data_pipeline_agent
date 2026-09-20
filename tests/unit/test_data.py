"""Unit tests for the data layer: profiler, schema comparison, quality, anomaly, lineage."""

from __future__ import annotations

import polars as pl
import pytest

from app.data.anomaly import detect_anomalies
from app.data.lineage import LineageEdge, LineageGraph
from app.data.profiler import profile_dataset, row_count
from app.data.quality import run_quality_checks
from app.data.schema import compare_schemas, compute_rename_hypotheses
from app.models import (
    ColumnDefinition,
    DataType,
    DriftEventType,
    SchemaDefinition,
)
from app.pipeline.schemas import infer_schema, polars_to_logical


def _orders_expected() -> SchemaDefinition:
    return SchemaDefinition(
        table="orders",
        version=3,
        columns=[
            ColumnDefinition(name="order_id", type=DataType.INTEGER),
            ColumnDefinition(name="amount", type=DataType.DOUBLE),
            ColumnDefinition(name="currency", type=DataType.VARCHAR),
        ],
    )


class TestProfiler:
    def test_profile_numeric(self):
        df = pl.DataFrame({"amount": [1.0, 2.0, 3.0]})
        p = profile_dataset(df)
        assert p["row_count"] == 3
        prof = p["profiles"]["amount"]
        assert prof["mean"] == 2.0
        assert prof["null_rate"] == 0.0
        assert prof["dtype"] == "Float64"

    def test_profiler_null_rate(self):
        df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
        p = profile_dataset(df)["profiles"]["a"]
        assert p["null_count"] == 2
        assert p["null_rate"] == pytest.approx(0.5)

    def test_row_count(self):
        assert row_count(pl.DataFrame({"x": [1, 2, 3, 4]})) == 4


class TestSchemaInference:
    def test_infer_types(self):
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "amount": [1.5, 2.5, 3.5],
                "name": ["a", "b", "c"],
                "ok": [True, False, True],
            }
        )
        sd = infer_schema(df, table="t")
        assert polars_to_logical(df["id"].dtype) == DataType.INTEGER
        types = {c.name: c.type for c in sd.columns}
        assert types["amount"] == DataType.DOUBLE
        assert types["name"] == DataType.VARCHAR
        assert types["ok"] == DataType.BOOLEAN


class TestSchemaComparison:
    def test_identical_no_events(self):
        expected = _orders_expected()
        observed = infer_schema(
            pl.DataFrame(
                {
                    "order_id": [1, 2],
                    "amount": [1.0, 2.0],
                    "currency": ["USD", "EUR"],
                }
            ),
            table="orders",
        )
        assert compare_schemas(expected, observed) == []

    def test_type_changed_high_severity(self):
        expected = _orders_expected()
        observed = SchemaDefinition(
            table="orders",
            version=4,
            columns=[
                ColumnDefinition(name="order_id", type=DataType.INTEGER),
                ColumnDefinition(name="amount", type=DataType.VARCHAR),
                ColumnDefinition(name="currency", type=DataType.VARCHAR),
            ],
        )
        events = compare_schemas(expected, observed)
        assert len(events) == 1
        e = events[0]
        assert e.type == DriftEventType.TYPE_CHANGED
        assert e.column == "amount"
        assert e.expected == "DOUBLE"
        assert e.observed == "VARCHAR"
        assert e.severity.value == "HIGH"

    def test_column_added_and_removed(self):
        expected = _orders_expected()
        observed = SchemaDefinition(
            table="orders",
            version=5,
            columns=[
                ColumnDefinition(name="order_id", type=DataType.INTEGER),
                ColumnDefinition(name="amount", type=DataType.DOUBLE),
                ColumnDefinition(name="customer_tier", type=DataType.VARCHAR),
            ],
        )
        events = {e.column: e.type for e in compare_schemas(expected, observed)}
        assert events["currency"] == DriftEventType.COLUMN_REMOVED
        assert events["customer_tier"] == DriftEventType.COLUMN_ADDED

    def test_rename_hypothesis(self):
        expected = _orders_expected()
        observed = SchemaDefinition(
            table="orders",
            version=6,
            columns=[
                ColumnDefinition(name="order_id", type=DataType.INTEGER),
                ColumnDefinition(name="amount", type=DataType.DOUBLE),
                ColumnDefinition(name="client_id", type=DataType.INTEGER),
            ],
        )
        # "currency" removed, "client_id" added -> product_id similarity only.
        hyps = compute_rename_hypotheses(expected, observed)
        assert isinstance(hyps, list)


class TestQuality:
    def test_null_spike_detected(self):
        df = pl.DataFrame({"amount": [1.0, None, None, None, None, 4.0, 5.0]})
        report, anomalies = run_quality_checks(df, pipeline_id="orders")
        failed = [c.name for c in report.failed]
        assert "null_rate.amount" in failed
        assert any(a.metric == "null_rate" and a.column == "amount" for a in anomalies)

    def test_duplicate_spike(self):
        df = pl.DataFrame({"id": [1, 1, 1, 1, 2, 3, 4]})
        report, _ = run_quality_checks(df, pipeline_id="orders")
        assert "duplicate_rate" in [c.name for c in report.failed]

    def test_clean_pass(self):
        df = pl.DataFrame({"id": [1, 2, 3, 4], "amount": [1.0, 2.0, 3.0, 4.0]})
        report, anomalies = run_quality_checks(df, pipeline_id="orders")
        assert report.passed is True
        assert anomalies == []


class TestAnomaly:
    def test_row_count_drop_anomaly(self):
        history = [
            {"timestamp": f"t{i}", "row_count": 100000 + i, "duplicate_rate": 0.001}
            for i in range(5)
        ]
        df = pl.DataFrame({"x": [1, 2, 3]})
        anomalies = detect_anomalies(df, history, previous_row_count=103000)
        assert any(a.metric == "row_count" for a in anomalies)

    def test_null_rate_spike_absolute_guard(self):
        history = [
            {"timestamp": f"t{i}", "row_count": 100, "null_rate.a": 0.01}
            for i in range(3)
        ]
        df = pl.DataFrame({"a": [None, None, None, None, 1.0, 2.0]})  # ~67% null
        anomalies = detect_anomalies(df, history)
        assert any(a.metric == "null_rate" and a.column == "a" for a in anomalies)

    def test_no_spurious_anomaly(self):
        history = [
            {"timestamp": f"t{i}", "row_count": 100 + i, "duplicate_rate": 0.001}
            for i in range(4)
        ]
        df = pl.DataFrame({"x": list(range(100))})
        anomalies = detect_anomalies(df, history)
        assert not any(a.metric == "row_count" for a in anomalies)


class TestLineage:
    def test_downstream(self):
        g = LineageGraph(
            [
                LineageEdge("raw.orders", "clean.orders", ["amount"]),
                LineageEdge("clean.orders", "daily_revenue", ["amount"]),
                LineageEdge("raw.customers", "clean.customers", ["id"]),
            ]
        )
        assert g.downstream("raw.orders") == ["clean.orders", "daily_revenue"]
        assert g.nodes == [
            "clean.customers",
            "clean.orders",
            "daily_revenue",
            "raw.customers",
            "raw.orders",
        ]

    def test_blast_radius_column_specific(self):
        g = LineageGraph(
            [
                LineageEdge("raw.orders", "clean.orders", ["amount", "currency"]),
                LineageEdge("clean.orders", "daily_revenue", ["amount"]),
            ]
        )
        radius = g.blast_radius(["amount"])
        assert "clean.orders" in radius
        assert "daily_revenue" in radius
        # A change to currency does NOT reach daily_revenue (not carried there)
        radius2 = g.blast_radius(["currency"])
        assert "daily_revenue" not in radius2
