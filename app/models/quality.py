"""Data quality metrics, checks, and quality anomaly models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.schema import Severity


class QualityCheck(BaseModel):
    """A single quality check definition + observed result."""

    name: str
    column: str | None = None
    metric: str  # e.g. "null_rate", "duplicate_rate", "row_count", "range"
    threshold: float | None = None
    observed: float | None = None
    status: str = "pending"  # pass | fail | skip
    message: str | None = None


class QualityReport(BaseModel):
    pipeline_id: str
    run_id: str | None = None
    checks: list[QualityCheck] = Field(default_factory=list)

    @property
    def failed(self) -> list[QualityCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def passed(self) -> bool:
        return all(c.status != "fail" for c in self.checks)


class QualityAnomaly(BaseModel):
    """A deterministic data-quality anomaly surfaced to the agent."""

    metric: str
    column: str | None = None
    expected: Any = None
    observed: Any = None
    severity: Severity = Severity.MEDIUM
    threshold: float | None = None
    context: str | None = None
