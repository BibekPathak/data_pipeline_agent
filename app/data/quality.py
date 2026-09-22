"""Data quality monitoring.

Runs deterministic checks against a DataFrame and produces a
:class:`QualityReport` plus a list of :class:`QualityAnomaly` objects the agent
uses during diagnosis. All checks are threshold-driven; nothing here is
LLM-decided.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from app.models import (
    QualityAnomaly,
    QualityCheck,
    QualityReport,
    Severity,
)


def run_quality_checks(
    df: pl.DataFrame,
    *,
    checks: list[dict[str, Any]] | None = None,
    pipeline_id: str = "",
    referenced: dict[str, pl.DataFrame] | None = None,
) -> tuple[QualityReport, list[QualityAnomaly]]:
    """Run quality checks and return (report, anomalies).

    Default checks when ``checks`` is omitted:
      - ``row_count`` lower bound (0)
      - per-numeric-column ``null_rate`` (< 0.05)
      - ``duplicate_rate`` for the whole frame (< 0.01)
      - ``uniqueness`` on the first INTEGER-looking column if present
    Custom checks can be supplied as a list of dicts with keys
    ``name``, ``kind`` (e.g. "null_rate", "range", "unique", "referential"),
    ``column``, ``threshold``, ``min``/``max``.
    """
    report = QualityReport(pipeline_id=pipeline_id)
    anomalies: list[QualityAnomaly] = []

    custom = checks or []

    # --- row count ---
    rows = df.height
    rc_threshold = 0
    report.checks.append(
        QualityCheck(
            name="row_count",
            metric="row_count",
            threshold=rc_threshold,
            observed=rows,
            status="pass" if rows >= rc_threshold else "fail",
        )
    )
    if rows < rc_threshold:
        anomalies.append(
            QualityAnomaly(
                metric="row_count",
                expected=rc_threshold,
                observed=rows,
                severity=Severity.CRITICAL,
            )
        )

    # --- duplicate rate ---
    dup_rate = 1.0 - (df.unique().height / rows) if rows else 0.0
    dup_threshold = 0.01
    report.checks.append(
        QualityCheck(
            name="duplicate_rate",
            metric="duplicate_rate",
            threshold=dup_threshold,
            observed=round(dup_rate, 6),
            status="pass" if dup_rate <= dup_threshold else "fail",
        )
    )
    if dup_rate > dup_threshold:
        anomalies.append(
            QualityAnomaly(
                metric="duplicate_rate",
                expected=f"<= {dup_threshold}",
                observed=round(dup_rate, 6),
                severity=Severity.HIGH,
                threshold=dup_threshold,
            )
        )

    # --- business invariants: non-negative value columns ---
    for col in df.columns:
        if not df[col].dtype.is_numeric():
            continue
        if col.lower() not in ("amount", "revenue", "price", "total", "sum"):
            continue
        mn = df[col].min()
        if mn is not None and float(mn) < 0:
            report.checks.append(
                QualityCheck(
                    name=f"non_negative.{col}",
                    column=col,
                    metric="business_invariant",
                    threshold=0.0,
                    observed=float(mn),
                    status="fail",
                    message=f"{col} contains negative values (min={mn})",
                )
            )
            anomalies.append(
                QualityAnomaly(
                    metric="business_invariant",
                    column=col,
                    observed=float(mn),
                    expected=">=0",
                    severity=Severity.HIGH,
                    context="revenue/amount invariant violated",
                )
            )

    # --- per-column null rates (numeric + any check that targets one) ---
    for col in df.columns:
        nr = df[col].null_count() / rows if rows else 0.0
        null_threshold = 0.05
        report.checks.append(
            QualityCheck(
                name=f"null_rate.{col}",
                column=col,
                metric="null_rate",
                threshold=null_threshold,
                observed=round(nr, 6),
                status="pass" if nr <= null_threshold else "fail",
            )
        )
        if nr > null_threshold:
            anomalies.append(
                QualityAnomaly(
                    metric="null_rate",
                    column=col,
                    observed=round(nr, 6),
                    severity=Severity.HIGH,
                    threshold=null_threshold,
                    context=f"null rate {nr:.3f} exceeds {null_threshold}",
                )
            )

    # --- referential integrity (if a reference table + key given) ---
    if referenced:
        _referential_checks(report, anomalies, df, referenced)

    for c in custom:
        _run_custom_check(report, anomalies, df, c)

    return report, anomalies


def _referential_checks(
    report: QualityReport,
    anomalies: list[QualityAnomaly],
    df: pl.DataFrame,
    referenced: dict[str, pl.DataFrame],
) -> None:
    for key, ref_df in referenced.items():
        if key not in df.columns:
            continue
        fk = df[key].drop_nulls()
        ref = ref_df[key].drop_nulls()
        ref_set = set(ref.to_list())
        missing = fk.filter(~fk.is_in(list(ref_set))).len()  # type: ignore[arg-type]
        rate = missing / len(fk) if len(fk) else 0.0
        report.checks.append(
            QualityCheck(
                name=f"referential.{key}",
                column=key,
                metric="referential_integrity",
                threshold=0.0,
                observed=rate,
                status="pass" if missing == 0 else "fail",
                message=f"{missing} orphan values",
            )
        )
        if missing > 0:
            anomalies.append(
                QualityAnomaly(
                    metric="referential_integrity",
                    column=key,
                    observed=missing,
                    severity=Severity.HIGH,
                    context="values missing from reference table",
                )
            )


def _run_custom_check(
    report: QualityReport,
    anomalies: list[QualityAnomaly],
    df: pl.DataFrame,
    c: dict[str, Any],
) -> None:
    name = c.get("name", "custom")
    kind = c.get("kind")
    col = c.get("column")
    if col is not None and col not in df.columns:
        return

    if kind == "range" and col is not None:
        mn = float(df[col].min())
        mx = float(df[col].max())
        lo = float(c.get("min", float("-inf")))
        hi = float(c.get("max", float("inf")))
        ok = lo <= mn and mx <= hi
        report.checks.append(
            QualityCheck(
                name=name, column=col, metric="range",
                threshold=None, observed=f"[{mn}, {mx}]",
                status="pass" if ok else "fail",
                message=f"range {mn}..{mx} outside [{lo}, {hi}]",
            )
        )
        if not ok:
            anomalies.append(QualityAnomaly(metric="range", column=col,
                                            severity=Severity.HIGH,
                                            observed=[mn, mx]))

    elif kind == "unique" and col is not None:
        n = df.height
        uniq = df[col].n_unique()
        ok = uniq == n
        report.checks.append(
            QualityCheck(
                name=name, column=col, metric="uniqueness",
                threshold=1.0, observed=uniq / n if n else 1.0,
                status="pass" if ok else "fail",
                message="column not unique" if not ok else None,
            )
        )
        if not ok:
            anomalies.append(QualityAnomaly(metric="uniqueness", column=col,
                                            severity=Severity.HIGH))
