"""Diagnosis engine: turn deterministic evidence into ranked hypotheses.

Hybrid approach: rule-based hypotheses are built from the drift events and
quality anomalies the system measured deterministically. The LLM then refines a
root-cause description and a fix proposal. The *selection* of a hypothesis is
still deterministic — the agent never lets the model pick a diagnosis blindly.
"""

from __future__ import annotations

import uuid

from app.models import (
    DriftEvent,
    DriftEventType,
    Hypothesis,
    QualityAnomaly,
)


class DiagnosisEngine:
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def build_hypotheses(
        self,
        drift_events: list[DriftEvent],
        quality_anomalies: list[QualityAnomaly],
        lineage_context: dict | None = None,
    ) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        lineage_context = lineage_context or {}

        drift_by_type: dict[DriftEventType, list[DriftEvent]] = {}
        for e in drift_events:
            drift_by_type.setdefault(e.type, []).append(e)

        if DriftEventType.TYPE_CHANGED in drift_by_type:
            cols = ", ".join(e.column for e in drift_by_type[DriftEventType.TYPE_CHANGED])
            hypotheses.append(
                Hypothesis(
                    id=_hid("type_drift"),
                    description=f"Upstream changed numeric-type columns to string: {cols}",
                    confidence=0.9,
                    evidence=[
                        f"{e.column}: {e.expected} -> {e.observed}"
                        for e in drift_by_type[DriftEventType.TYPE_CHANGED]
                    ],
                )
            )

        if DriftEventType.COLUMN_REMOVED in drift_by_type:
            cols = ", ".join(e.column for e in drift_by_type[DriftEventType.COLUMN_REMOVED])
            hypotheses.append(
                Hypothesis(
                    id=_hid("column_removed"),
                    description=f"Required columns disappeared: {cols}",
                    confidence=0.85,
                    evidence=[f"missing {cols}"],
                )
            )

        if DriftEventType.COLUMN_ADDED in drift_by_type:
            cols = ", ".join(e.column for e in drift_by_type[DriftEventType.COLUMN_ADDED])
            hypotheses.append(
                Hypothesis(
                    id=_hid("column_added"),
                    description=f"New non-breaking columns present: {cols}",
                    confidence=0.7,
                    evidence=[f"added {cols}"],
                )
            )

        # Quality-driven hypotheses.
        metrics = [a.metric for a in quality_anomalies]
        if "duplicate_rate" in metrics:
            hypotheses.append(
                Hypothesis(
                    id=_hid("duplicates"),
                    description="Duplicate records detected above threshold",
                    confidence=0.85,
                    evidence=["duplicate_rate anomaly"],
                )
            )
        if "null_rate" in metrics:
            hypotheses.append(
                Hypothesis(
                    id=_hid("nulls"),
                    description="Null rates spiked on one or more columns",
                    confidence=0.8,
                    evidence=[
                        f"{a.column}={a.observed}" for a in quality_anomalies if a.metric == "null_rate"
                    ],
                )
            )
        if "row_count" in metrics:
            hypotheses.append(
                Hypothesis(
                    id=_hid("volume"),
                    description="Row-count anomaly (possible truncation/dup ingestion)",
                    confidence=0.75,
                    evidence=[
                        f"observed={a.observed} expected={a.expected}"
                        for a in quality_anomalies if a.metric == "row_count"
                    ],
                )
            )

        # Sort by confidence.
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def select_hypothesis(
        self, hypotheses: list[Hypothesis], selected_id: str | None = None
    ) -> Hypothesis | None:
        if not hypotheses:
            return None
        if selected_id is not None:
            for h in hypotheses:
                if h.id == selected_id:
                    h.is_selected = True
                    return h
        best = hypotheses[0]
        best.is_selected = True
        return best


def _hid(kind: str) -> str:
    return f"{kind}:{uuid.uuid4().hex[:6]}"
