"""Metric repository for pipeline observability.

Records a snapshot of key metrics per table/timeframe into the metadata store,
and reconstructs a history that feeds deterministic anomaly detection (see
``app/data/anomaly.py``).

The metric names follow a flat dotted convention so history is easy to aggregate:
  - ``row_count``
  - ``duplicate_rate``
  - ``mean.<column>``  (numeric columns)
  - ``null_rate.<column>``
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl
from app.storage.base import MetadataStore


class MetricRepository:
    def __init__(self, metadata: MetadataStore) -> None:
        self._metadata = metadata

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def snapshot(self, df: pl.DataFrame) -> dict[str, float]:
        """Compute the metric snapshot for a DataFrame."""
        rows = df.height
        metrics: dict[str, float] = {"row_count": float(rows)}
        if rows:
            metrics["duplicate_rate"] = 1.0 - (df.unique().height / rows)
        for col in df.columns:
            nr = df[col].null_count() / rows if rows else 0.0
            metrics[f"null_rate.{col}"] = round(nr, 6)
            if df[col].dtype.is_numeric() and rows:
                metrics[f"mean.{col}"] = round(float(df[col].mean()), 6)
                metrics[f"min.{col}"] = float(df[col].min())
                metrics[f"max.{col}"] = float(df[col].max())
        return metrics

    async def record(self, table: str, df: pl.DataFrame, at: str | None = None) -> None:
        """Record a snapshot for ``table``."""
        metrics = self.snapshot(df)
        await self._metadata.record_metrics(
            table, at or self._now(), metrics
        )

    async def history(self, table: str) -> list[dict[str, Any]]:
        """Return historical snapshots for ``table`` (each with a timestamp)."""
        return await self._metadata.get_metric_history(table)
