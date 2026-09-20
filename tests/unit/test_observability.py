"""Unit tests for the observability layer (metrics repository + health monitor)."""

from __future__ import annotations

import polars as pl
import pytest

from app.observability.metrics import MetricRepository
from app.observability.monitor import HealthMonitor
from app.storage import MemoryMetadataStore


def _df(rows: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {"order_id": list(range(rows)), "amount": [i * 1.5 for i in range(rows)]}
    )


class TestMetricRepository:
    @pytest.mark.asyncio
    async def test_snapshot_fields(self):
        repo = MetricRepository(MemoryMetadataStore())
        s = repo.snapshot(_df(100))
        assert s["row_count"] == 100.0
        assert "duplicate_rate" in s
        assert s["duplicate_rate"] == pytest.approx(0.0)
        assert "mean.amount" in s
        assert "null_rate.amount" in s
        assert s["null_rate.amount"] == 0.0

    @pytest.mark.asyncio
    async def test_record_and_history(self):
        meta = MemoryMetadataStore()
        repo = MetricRepository(meta)
        await repo.record("orders", _df(100), at="t1")
        await repo.record("orders", _df(90), at="t2")
        await repo.record("other", _df(50), at="t1")
        hist = await repo.history("orders")
        assert len(hist) == 2
        assert hist[0]["row_count"] == 100.0
        assert hist[1]["row_count"] == 90.0

    @pytest.mark.asyncio
    async def test_null_rate_detected(self):
        meta = MemoryMetadataStore()
        repo = MetricRepository(meta)
        df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
        await repo.record("t", df, at="t1")
        hist = await repo.history("t")
        assert hist[0]["null_rate.a"] == pytest.approx(0.5)


class TestHealthMonitor:
    def test_healthy_canary(self):
        m = HealthMonitor()
        r = m.evaluate_canary(
            {"failure_rate": 0.0, "quality_pass": True, "status": "CANARY_PASSED"}
        )
        assert r.healthy is True

    def test_unhealthy_on_failure(self):
        m = HealthMonitor()
        r = m.evaluate_canary({"failure_rate": 1.0, "quality_pass": True})
        assert r.healthy is False
        assert any(c.name == "failure_rate" and not c.passed for c in r.checks)

    def test_unhealthy_on_quality(self):
        m = HealthMonitor()
        r = m.evaluate_canary({"failure_rate": 0.0, "quality_pass": False})
        assert r.healthy is False

    def test_baseline_anomalies(self):
        m = HealthMonitor()
        history = [
            {"timestamp": f"t{i}", "row_count": 100000 + i} for i in range(5)
        ]
        anomalies = m.baseline_anomalies(pl.DataFrame({"x": [1, 2, 3]}), history)
        assert any(a.metric == "row_count" for a in anomalies)
