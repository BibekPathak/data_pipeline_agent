"""Shadow validation engine.

Compares the *current* pipeline output against the *candidate* (fixed) output on
the same observed data, producing a :class:`ValidationReport`. Used before any
deployment.

Two cases:
  - Current is healthy         -> delta checks (row count, aggregates, quality).
  - Current is broken (drift)  -> validate candidate against the pipeline's
    declared output schema and absolute invariants (schema, quality, no data
    loss vs input row count, non-negative business metrics).

The candidate must never be allowed to silently destroy data or corrupt metrics.
"""

from __future__ import annotations

import polars as pl

from app.models import SchemaDefinition, ValidationCheck, ValidationReport, ValidationStatus
from app.pipeline.runner import RunResult
from app.pipeline.schemas import infer_schema

# Conventional business-metric aliases: aggregated output column -> source
# input column. Used to compare a candidate's business metric against the raw
# input even when the current pipeline is broken (no healthy baseline).
BUSINESS_ALIASES = {"revenue": "amount", "total_revenue": "amount", "total": "amount"}


def _add(report: ValidationReport, name: str, ok: bool, **kw) -> None:
    report.checks.append(
        ValidationCheck(
            name=name,
            status=ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
            **kw,
        )
    )


def shadow_validate(
    current: RunResult,
    candidate: RunResult,
    *,
    pipeline_id: str = "",
    run_id: str = "",
    expected_schema: SchemaDefinition | None = None,
    input_row_count: int | None = None,
    constraints: dict | None = None,
    known_good: pl.DataFrame | None = None,
    expected_aggregates: dict[str, float] | None = None,
    input_df: pl.DataFrame | None = None,
) -> ValidationReport:
    report = ValidationReport(run_id=run_id)
    cand = candidate.final_df if candidate.ok else None
    cur = current.final_df if current.ok else None

    if cand is None:
        report.checks.append(
            ValidationCheck(name="candidate_output", status=ValidationStatus.FAILED,
                            message="candidate pipeline produced no output")
        )
        return report

    # --- candidate must match the expected output schema (if provided) ---
    if expected_schema is not None:
        inferred = infer_schema(cand, table=expected_schema.table)
        from app.data.schema import compare_schemas

        mismatches = compare_schemas(expected_schema, inferred)
        _add(report, "schema", not mismatches,
             expected=expected_schema.model_dump(), observed=inferred.model_dump(),
             message=f"schema mismatches: {[m.column for m in mismatches]}")
    else:
        _add(report, "schema", True)

    # --- data-loss guard vs input rows ---
    if input_row_count is not None:
        loss_ok = cand.height >= input_row_count * 0.95
        _add(report, "row_count_preserved", loss_ok,
             expected=input_row_count, observed=cand.height,
             message=f"candidate dropped to {cand.height} from {input_row_count}")

    # --- quality invariants on candidate ---
    from app.data.quality import run_quality_checks

    quality_report, _ = run_quality_checks(cand, pipeline_id=pipeline_id)
    _add(report, "quality", quality_report.passed,
         message=str([c.name for c in quality_report.failed]) or None)

    # --- business invariants (non-negative metrics) passed via candidate columns ---
    for col in cand.columns:
        dtype = cand[col].dtype
        if dtype.is_numeric() and col.lower() in (
            "revenue", "amount", "price", "total", "sum",
        ):
            mn = float(cand[col].min())
            _add(report, f"invariant.{col}", mn >= 0 or mn is None,
                 expected=">=0", observed=mn)

    # --- explicit constraints (uniqueness / range) ---
    constraints = constraints or {}
    unique_cols = constraints.get("unique", [])
    for col in unique_cols:
        if col not in cand.columns:
            continue
        uniq = cand[col].n_unique() == cand.height
        _add(report, f"unique.{col}", uniq,
             expected="unique", observed=f"{cand[col].n_unique()}/{cand.height}",
             message=f"{col} not unique" if not uniq else None)
    for col, bounds in (constraints.get("range", {}) or {}).items():
        if col not in cand.columns or not cand[col].dtype.is_numeric():
            continue
        mn = float(cand[col].min())
        mx = float(cand[col].max())
        lo = float(bounds[0])
        hi = float(bounds[1])
        ok = lo <= mn and mx <= hi
        _add(report, f"range.{col}", ok, expected=f"[{lo},{hi}]", observed=f"[{mn},{mx}]",
             message=f"{col} out of range" if not ok else None)

    # --- regression vs known-good historical output ---
    if known_good is not None:
        if cand.height != known_good.height:
            _add(report, "regression.row_count", False,
                 expected=known_good.height, observed=cand.height,
                 message="row count differs from known-good output")
        else:
            _add(report, "regression.row_count", True,
                 expected=known_good.height, observed=cand.height)

    # --- expected aggregates (business-metric baseline from input data) ---
    # When the current pipeline is broken (no healthy baseline to delta against),
    # the candidate is still compared against aggregates computed from the raw
    # input, so silent corruption cannot slip through. Handles both row-level
    # outputs (same column name) and aggregated outputs (revenue <- amount).
    expected_checks: list[tuple[str, float, float]] = []  # (name, expected, observed)

    for col, expected_sum in (expected_aggregates or {}).items():
        if col in cand.columns and cand[col].dtype.is_numeric():
            expected_checks.append((col, expected_sum, float(cand[col].sum())))

    if input_df is not None:
        for out_col, in_col in BUSINESS_ALIASES.items():
            if out_col not in cand.columns or not cand[out_col].dtype.is_numeric():
                continue
            if in_col not in input_df.columns:
                continue
            src = input_df[in_col].cast(pl.Float64, strict=False)
            expected_sum = float(src.sum())
            expected_checks.append((out_col, expected_sum, float(cand[out_col].sum())))

    for name, expected_sum, observed_sum in expected_checks:
        if expected_sum != 0 and abs(observed_sum - expected_sum) / abs(expected_sum) > 0.05:
            _add(report, f"expected_aggregate.{name}", False,
                 expected=expected_sum, observed=observed_sum,
                 message=f"{name} sum diverges from input baseline by >5%")
        else:
            _add(report, f"expected_aggregate.{name}", True,
                 expected=expected_sum, observed=observed_sum)

    # --- delta comparison when current is healthy ---
    if cur is not None:
        cur_rows = cur.height
        cand_rows = cand.height
        if cur_rows and cand_rows < cur_rows * 0.95:
            _add(report, "row_count_vs_current", False,
                 expected=cur_rows, observed=cand_rows,
                 message=f"candidate lost >5% vs current ({cur_rows}->{cand_rows})")
        else:
            _add(report, "row_count_vs_current", True, expected=cur_rows, observed=cand_rows)

        for col in cur.columns:
            if not cur[col].dtype.is_numeric() or col not in cand.columns:
                continue
            s_cur = float(cur[col].sum())
            s_cand = float(cand[col].sum())
            if s_cur != 0 and abs(s_cand - s_cur) / abs(s_cur) > 0.05:
                _add(report, f"aggregate.{col}", False, expected=s_cur, observed=s_cand,
                     message=f"{col} sum changed >5%")
            elif s_cur != 0:
                _add(report, f"aggregate.{col}", True, expected=s_cur, observed=s_cand)

    return report
