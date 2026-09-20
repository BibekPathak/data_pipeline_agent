"""Dataset profiling: compute deterministic descriptive statistics.

Profiles feed both data quality checks and anomaly detection. Everything here
is computed from the data directly (no LLM involved).
"""

from __future__ import annotations

from typing import Any

import polars as pl


class ProfileResult:
    """Parsed profile for a single column."""

    def __init__(
        self,
        column: str,
        dtype: str,
        null_count: int,
        null_rate: float,
        distinct_count: int,
        min: Any = None,
        max: Any = None,
        mean: float | None = None,
        std: float | None = None,
    ) -> None:
        self.column = column
        self.dtype = dtype
        self.null_count = null_count
        self.null_rate = null_rate
        self.distinct_count = distinct_count
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "null_rate": self.null_rate,
            "distinct_count": self.distinct_count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
        }


def _col_profile(df: pl.DataFrame, col: str) -> ProfileResult:
    series = df[col]
    total = df.height
    null_count = series.null_count()
    null_rate = (null_count / total) if total else 0.0
    distinct_count = series.n_unique()

    try:
        min_v = series.min()
        max_v = series.max()
    except Exception:  # non-orderable types
        min_v = max_v = None

    mean = std = None
    if series.dtype.is_numeric():
        try:
            mean = float(series.mean())
            std = float(series.std())
        except Exception:
            pass

    return ProfileResult(
        column=col,
        dtype=str(series.dtype),
        null_count=null_count,
        null_rate=null_rate,
        distinct_count=distinct_count,
        min=min_v,
        max=max_v,
        mean=mean,
        std=std,
    )


def profile_dataset(df: pl.DataFrame) -> dict[str, Any]:
    """Return a full profile: row count + per-column ProfileResult dicts."""
    profiles = {col: _col_profile(df, col) for col in df.columns}
    return {
        "row_count": df.height,
        "columns": df.columns,
        "profiles": {c: p.to_dict() for c, p in profiles.items()},
    }


def row_count(df: pl.DataFrame) -> int:
    return df.height


def value_counts(df: pl.DataFrame, col: str, top: int = 10) -> list[dict[str, Any]]:
    """Type/value distribution for a column (for category-drift detection)."""
    vc = (
        df[col]
        .cast(pl.String, strict=False)
        .drop_nulls()
        .value_counts(sort=True)
        .head(top)
    )
    return [
        {"value": row[col], "count": row["count"]}
        for row in vc.iter_rows(named=True)
    ]
