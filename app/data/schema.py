"""Schema comparison engine: deterministic drift detection.

Detects additions, removals, type changes, nullability changes and produces
*rename hypotheses* (never treated as fact) using name/type/value similarity.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import polars as pl

from app.models import (
    DriftEvent,
    DriftEventType,
    RenameHypothesis,
    SchemaDefinition,
    Severity,
)


def compare_schemas(
    expected: SchemaDefinition, observed: SchemaDefinition
) -> list[DriftEvent]:
    """Compare expected vs observed, returning a list of drift events."""
    exp_map = expected.column_map
    obs_map = observed.column_map
    exp_names = set(exp_map)
    obs_names = set(obs_map)

    events: list[DriftEvent] = []

    for name in sorted(obs_names - exp_names):
        events.append(
            DriftEvent(
                type=DriftEventType.COLUMN_ADDED,
                column=name,
                observed=obs_map[name].type.value,
                severity=Severity.LOW,
            )
        )

    for name in sorted(exp_names - obs_names):
        events.append(
            DriftEvent(
                type=DriftEventType.COLUMN_REMOVED,
                column=name,
                expected=exp_map[name].type.value,
                severity=Severity.MEDIUM,
            )
        )

    for name in sorted(exp_names & obs_names):
        e = exp_map[name]
        o = obs_map[name]
        if e.type != o.type:
            events.append(
                DriftEvent(
                    type=DriftEventType.TYPE_CHANGED,
                    column=name,
                    expected=e.type.value,
                    observed=o.type.value,
                    severity=Severity.HIGH,
                )
            )
        if e.nullable != o.nullable:
            events.append(
                DriftEvent(
                    type=DriftEventType.NULLABLE_CHANGED,
                    column=name,
                    expected=str(e.nullable),
                    observed=str(o.nullable),
                    severity=Severity.MEDIUM,
                )
            )

    return events


def compute_rename_hypotheses(
    expected: SchemaDefinition,
    observed: SchemaDefinition,
    df: pl.DataFrame | None = None,
    *,
    name_weight: float = 0.4,
    type_weight: float = 0.2,
    value_weight: float = 0.4,
    threshold: float = 0.6,
) -> list[RenameHypothesis]:
    """Suggest possible renames for removed/added column pairs.

    Uses name similarity (difflib), type similarity (identical logical type),
    and value-statistics similarity (overlap of the sorted distinct value sets).
    Results are *hypotheses*, not facts.
    """
    exp_map = expected.column_map
    obs_map = observed.column_map
    exp_names = set(exp_map)
    obs_names = set(obs_map)
    removed = sorted(exp_names - obs_names)
    added = sorted(obs_names - exp_names)

    def has_values() -> bool:
        return df is not None and all(c in df.columns for c in removed + added)

    def value_similarity(a: str, b: str) -> float:
        if df is None or a not in df.columns or b not in df.columns:
            return 0.0
        sa = {
            str(v)
            for v in df[a].cast(pl.String, strict=False).drop_nulls().unique().to_list()
        }
        sb = {
            str(v)
            for v in df[b].cast(pl.String, strict=False).drop_nulls().unique().to_list()
        }
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(len(sa), len(sb))

    hypotheses: list[RenameHypothesis] = []
    for r in removed:
        for a in added:
            name_sim = SequenceMatcher(None, r, a).ratio()
            type_sim = 1.0 if exp_map[r].type == obs_map[a].type else 0.0
            val_sim = value_similarity(r, a) if has_values() else 0.0
            confidence = (
                name_weight * name_sim
                + type_weight * type_sim
                + value_weight * val_sim
            )
            if confidence >= threshold:
                hypotheses.append(
                    RenameHypothesis(
                        source=r,
                        target=a,
                        name_similarity=round(name_sim, 3),
                        type_similarity=round(type_sim, 3),
                        value_similarity=round(val_sim, 3),
                        historical_support=False,
                        confidence=round(confidence, 3),
                    )
                )
    return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
