"""Benchmark scenarios: 10 deterministic failures the agent must handle.

Each scenario's ``build`` function turns the healthy orders DataFrame into a
specific failure. Scenarios relying on baseline anomaly detection expose a
``seed_metrics`` async callback that the runner invokes to seed metric history
*before* the run.

Expected outcomes reflect the deterministic, conservative agent:
  - safe coercions (cast/parse/fill)   -> FIXED (auto-approve LOW risk)
  - uncertain/among benign changes     -> NO_ACTION (no destructive mutation)
  - destructive/needs-approval fixes   -> NO_ACTION or POLICY_REJECTED
"""

from __future__ import annotations

import random

import polars as pl

from app.evaluation.models import ExpectedOutcome, Scenario, ScenarioKind


def _R(seed: int) -> random.Random:
    return random.Random(seed)


def _healthy() -> pl.DataFrame:
    from app.data.loader import load_source

    return load_source("fixtures:orders.csv", "./datasets/fixtures")


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="type_drift",
            name="Amount type drift DOUBLE -> VARCHAR",
            category="schema",
            kind=ScenarioKind.HEALING,
            build=lambda df: df.with_columns(
                pl.col("amount").cast(pl.String).alias("amount")
            ),
            expected_diagnosis=["amount", "type"],
            acceptable_fix=["cast_type"],
            expected_outcome=ExpectedOutcome.FIXED,
            description="Numeric amount became a string column; agent must cast safely.",
        ),
        Scenario(
            id="added_nullable_column",
            name="New nullable column (customer_tier)",
            category="schema",
            kind=ScenarioKind.INVESTIGATE,
            build=lambda df: df.with_columns(
                pl.lit(None).cast(pl.String).alias("customer_tier")
            ),
            expected_diagnosis=["customer_tier", "column_added"],
            acceptable_fix=[],
            expected_outcome=ExpectedOutcome.NO_ACTION,
            description="Non-breaking column addition; agent must tolerate, not mutate.",
        ),
        Scenario(
            id="column_rename",
            name="Column rename (currency -> currency_code)",
            category="schema",
            kind=ScenarioKind.INVESTIGATE,
            build=lambda df: df.rename({"currency": "currency_code"}),
            expected_diagnosis=["currency_code", "rename"],
            acceptable_fix=["rename_column"],
            expected_outcome=ExpectedOutcome.NO_ACTION,
            description="Possible rename is a hypothesis; agent must not auto-mutate.",
        ),
        Scenario(
            id="timestamp_format",
            name="Timestamp as ISO string (created_at)",
            category="schema",
            kind=ScenarioKind.HEALING,
            build=lambda df: df.with_columns(
                pl.col("created_at").cast(pl.String).alias("created_at")
            ),
            expected_diagnosis=["created_at", "type"],
            acceptable_fix=["cast_type"],
            expected_outcome=ExpectedOutcome.FIXED,
            description="created_at arrived as string; agent must parse it back.",
        ),
        Scenario(
            id="null_spike",
            name="Null spike on amount (~22%)",
            category="quality",
            kind=ScenarioKind.HEALING,
            build=lambda df: _inject_nulls(df, "amount", 0.22, seed=5),
            seed_metrics=_mk_seed_row_count(600),
            expected_diagnosis=["null_rate", "amount"],
            acceptable_fix=["default_value", "fill_null"],
            expected_outcome=ExpectedOutcome.FIXED,
            description="amount nulls spike; agent should fill with a safe default.",
        ),
        Scenario(
            id="duplicate_spike",
            name="Duplicate spike (~14% rows)",
            category="quality",
            kind=ScenarioKind.INVESTIGATE,
            build=lambda df: _inject_duplicates(df, 0.14, seed=6),
            seed_metrics=_mk_seed_row_count(600),
            expected_diagnosis=["duplicate_rate"],
            acceptable_fix=["deduplicate"],
            expected_outcome=ExpectedOutcome.NO_ACTION,
            description="Dedup is MEDIUM-risk so requires explicit approval; agent waits.",
        ),
        Scenario(
            id="invalid_numeric",
            name="Invalid numeric tokens ('unknown','NaN','')",
            category="quality",
            kind=ScenarioKind.HEALING,
            build=lambda df: _inject_invalid_numeric(df, seed=7),
            expected_diagnosis=["amount", "type"],
            acceptable_fix=["cast_type"],
            expected_outcome=ExpectedOutcome.FIXED,
            description="amount mixes numeric tokens; agent must cast safely.",
        ),
        Scenario(
            id="row_count_anomaly",
            name="Row count anomaly (600 -> 60)",
            category="volume",
            kind=ScenarioKind.INVESTIGATE,
            build=lambda df: df.head(df.height // 10),
            seed_metrics=_mk_seed_row_count(600, n=4),
            expected_diagnosis=["row_count"],
            acceptable_fix=[],
            expected_outcome=ExpectedOutcome.NO_ACTION,
            description="Volume drop; agent investigates, never fabricates or deletes rows.",
        ),
        Scenario(
            id="referential_integrity",
            name="Referential integrity break (orphan customer_id)",
            category="quality",
            kind=ScenarioKind.BLOCKED,
            build=lambda df: df.with_columns(
                (pl.col("customer_id") + 10000).alias("customer_id")
            ),
            expected_diagnosis=["referential_integrity", "customer_id"],
            acceptable_fix=["drop_invalid_rows"],
            expected_outcome=ExpectedOutcome.POLICY_REJECTED,
            description="Orphan references; deleting rows is HIGH-risk and must be gated.",
        ),
        Scenario(
            id="transformation_regression",
            name="Transformation regression (negative revenue category)",
            category="business",
            kind=ScenarioKind.INVESTIGATE,
            build=lambda df: df.with_columns(
                pl.when(pl.col("currency") == "EUR")
                .then(-pl.col("amount"))
                .otherwise(pl.col("amount"))
                .alias("amount")
            ),
            expected_diagnosis=["amount", "business_invariant"],
            acceptable_fix=[],
            expected_outcome=ExpectedOutcome.NO_ACTION,
            description="A category now yields negative revenue; agent surfaces it and never silently clamps.",
        ),
    ]


def _inject_nulls(df: pl.DataFrame, col: str, rate: float, seed: int) -> pl.DataFrame:
    rng = _R(seed)
    mask = [rng.random() < rate for _ in range(df.height)]
    values = df[col].to_list()
    for i, m in enumerate(mask):
        if m:
            values[i] = None
    return df.with_columns(pl.Series(col, values).alias(col))


def _inject_duplicates(df: pl.DataFrame, rate: float, seed: int) -> pl.DataFrame:
    rng = _R(seed)
    n_dups = int(df.height * rate)
    # Duplicate rows VERBATIM (same order_id) so full-row duplicate detection
    # and key-uniqueness checks both fire.
    sampled = df.sample(n=n_dups, seed=seed)
    return pl.concat([df, sampled])


def _inject_invalid_numeric(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = _R(seed)
    tokens = ["unknown", "NaN", "", "N/A"]
    values = [str(v) for v in df["amount"].to_list()]
    for i in range(len(values)):
        if rng.random() < 0.05:
            values[i] = rng.choice(tokens)
    return df.with_columns(pl.Series("amount", values).alias("amount"))


def _mk_seed_row_count(row_count: int, n: int = 4) -> callable:
    """Seed prior metric history so anomaly detection has a baseline."""

    async def seed(meta) -> None:
        for i in range(n):
            await meta.record_metrics(
                "orders", f"baseline-{i}",
                {"row_count": float(row_count + i), "duplicate_rate": 0.001},
            )

    return seed
