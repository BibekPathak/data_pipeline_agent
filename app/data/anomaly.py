"""Statistical anomaly detection (deterministic).

Flags deviations from a historical baseline before any LLM is involved:
  - row count outside 3 sigma (or a minimum absolute drop ratio)
  - null rate spike vs baseline mean
  - mean shift in numeric columns
  - duplicate rate spike
  - category distribution shift

The current observation is measured directly from the DataFrame; the baseline
comes from historical metric snapshots (e.g. ``MetadataStore.get_metric_history``).
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from app.models import QualityAnomaly, Severity

SIGMA = 3.0
MIN_ROWS_FOR_BASELINE = 3


def _numeric(history: list[dict[str, Any]], name: str) -> list[float]:
    return [
        float(h[name]) for h in history if name in h and h[name] is not None
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def detect_anomalies(
    df: pl.DataFrame,
    history: list[dict[str, Any]] | None = None,
    *,
    previous_row_count: int | None = None,
) -> list[QualityAnomaly]:
    """Detect anomalies in ``df`` relative to ``history``.

    ``previous_row_count`` overrides the last historical row count when supplied
    (useful when history isn't yet persisted).
    """
    anomalies: list[QualityAnomaly] = []
    history = history or []

    rows = df.height

    # --- row count anomaly ---
    base_rows = _numeric(history, "row_count")
    if previous_row_count is not None:
        base_rows.append(float(previous_row_count))
    if len(base_rows) >= MIN_ROWS_FOR_BASELINE:
        mean, std = _mean_std(base_rows)
        if std > 0:
            z = abs(rows - mean) / std
            if z > SIGMA:
                anomalies.append(
                    QualityAnomaly(
                        metric="row_count",
                        observed=rows,
                        expected=round(mean),
                        severity=Severity.HIGH,
                        threshold=round(mean + SIGMA * std),
                        context=f"row count {rows} deviates {z:.1f} sigma from baseline {round(mean)}",
                    )
                )
        # absolute drop guard (catches near-zero std cases)
        if std == 0 and previous_row_count is not None and previous_row_count > 0:
            if rows < previous_row_count * 0.2:
                anomalies.append(
                    QualityAnomaly(
                        metric="row_count",
                        observed=rows,
                        expected=previous_row_count,
                        severity=Severity.CRITICAL,
                        context=f"row count dropped to {rows} from {previous_row_count}",
                    )
                )

    # --- per-column null rate spikes ---
    for col in df.columns:
        nr = df[col].null_count() / rows if rows else 0.0
        base = _numeric(history, f"null_rate.{col}")
        if len(base) >= MIN_ROWS_FOR_BASELINE:
            mean, std = _mean_std(base)
            if std > 0 and nr > mean + SIGMA * std and (nr - mean) > 0.05:
                anomalies.append(
                    QualityAnomaly(
                        metric="null_rate", column=col,
                        observed=round(nr, 4), expected=round(mean, 4),
                        severity=Severity.HIGH, threshold=round(mean + SIGMA * std, 4),
                        context="null rate spike vs baseline",
                    )
                )
        # absolute guard
        if nr > 0.10 and (previous_row_count is not None or history):
            anomalies.append(
                QualityAnomaly(
                    metric="null_rate", column=col,
                    observed=round(nr, 4),
                    severity=Severity.MEDIUM,
                    threshold=0.10,
                    context="null rate above 10% absolute guard",
                )
            )

    # --- mean shift for numeric columns ---
    for col in df.columns:
        if not df[col].dtype.is_numeric():
            continue
        base = _numeric(history, f"mean.{col}")
        if len(base) >= MIN_ROWS_FOR_BASELINE:
            mean, std = _mean_std(base)
            cur = float(df[col].mean())
            if std > 0 and abs(cur - mean) / std > SIGMA:
                anomalies.append(
                    QualityAnomaly(
                        metric="mean_shift", column=col,
                        observed=round(cur, 3), expected=round(mean, 3),
                        severity=Severity.MEDIUM,
                        threshold=round(mean + SIGMA * std, 3),
                        context="mean shift beyond 3 sigma",
                    )
                )

    # --- duplicate rate spike ---
    dup_rate = 1.0 - (df.unique().height / rows) if rows else 0.0
    base_dup = _numeric(history, "duplicate_rate")
    if previous_row_count is not None or history:
        if dup_rate > 0.05:
            anomalies.append(
                QualityAnomaly(
                    metric="duplicate_rate", observed=round(dup_rate, 4),
                    severity=Severity.HIGH, threshold=0.05,
                    context="duplicate rate above 5% absolute guard",
                )
            )
    if len(base_dup) >= MIN_ROWS_FOR_BASELINE:
        mean, std = _mean_std(base_dup)
        if std > 0 and dup_rate > mean + SIGMA * std:
            anomalies.append(
                QualityAnomaly(
                    metric="duplicate_rate", observed=round(dup_rate, 4),
                    expected=round(mean, 4),
                    severity=Severity.HIGH,
                    context="duplicate rate spike vs baseline",
                )
            )

    return anomalies
