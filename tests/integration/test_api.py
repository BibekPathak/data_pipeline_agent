"""Integration tests for the minimal FastAPI surface."""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from app.config import StorageBackend

    monkeypatch.setattr(settings, "storage_backend", StorageBackend.MEMORY)
    monkeypatch.setattr(settings, "db_path", tmp_path / "api.db")
    from app.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["llm_provider"] == "deterministic"
        assert body["storage_backend"] == "memory"


class TestPipelines:
    def test_list_pipelines(self, client):
        r = client.get("/pipelines")
        assert r.status_code == 200
        pipelines = r.json()
        assert any(p["id"] == "orders" for p in pipelines)
        orders = next(p for p in pipelines if p["id"] == "orders")
        assert "orders_clean" in orders["stages"]


class TestTriage:
    def test_triage_healthy_data_no_action(self, client):
        r = client.post("/pipelines/orders/triage")
        assert r.status_code == 200
        body = r.json()
        assert body["phase"] == "SAFE_STOP"
        assert body["deployment_status"] == "not_deployed"
        assert body["run_id"]

    def test_triage_drifted_source_fixes(self, client, tmp_path):
        # Drift must survive the CSV round-trip: polars re-infers numeric
        # columns on read, so inject non-numeric tokens to force a String
        # column (type_changed drift) that the agent must repair.
        df = pl.read_csv("./datasets/fixtures/orders.csv")
        values = [str(v) for v in df["amount"].to_list()]
        values[0] = "unknown"
        values[1] = "NaN"
        drifted = df.with_columns(pl.Series("amount", values))
        path = tmp_path / "orders_drifted.csv"
        drifted.write_csv(path)

        r = client.post("/pipelines/orders/triage", json={"source": str(path)})
        assert r.status_code == 200
        body = r.json()
        assert body["phase"] == "SUCCESS"
        assert body["deployment_status"] == "active"
        assert body["proposed_fix"] is not None
        ops = [o["operation"] for o in body["proposed_fix"]["operations"]]
        assert "cast_type" in ops

    def test_triage_row_limit(self, client):
        r = client.post("/pipelines/orders/triage", json={"limit": 50})
        assert r.status_code == 200
        assert r.json()["run_id"]

    def test_unknown_pipeline_404(self, client):
        r = client.post("/pipelines/nope/triage")
        assert r.status_code == 404

    def test_bad_source_400(self, client):
        r = client.post("/pipelines/orders/triage", json={"source": "./missing.csv"})
        assert r.status_code == 400


class TestRuns:
    def test_get_run_after_triage(self, client):
        created = client.post("/pipelines/orders/triage").json()
        r = client.get(f"/runs/{created['run_id']}")
        assert r.status_code == 200
        assert r.json()["run_id"] == created["run_id"]
        assert r.json()["pipeline_id"] == "orders"

    def test_unknown_run_404(self, client):
        r = client.get("/runs/does-not-exist")
        assert r.status_code == 404
