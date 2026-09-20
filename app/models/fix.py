"""Declarative fix proposals, safe transformations, and validation reports."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FixOperationType(str, Enum):
    """The constrained set of safe transformations the engine may apply.

    Arbitrary SQL is deliberately NOT supported. Each operation is declarative
    and translated into a controlled Polars transformation by the engine.
    """

    CAST_TYPE = "cast_type"
    RENAME_COLUMN = "rename_column"
    FILL_NULL = "fill_null"
    DROP_INVALID_ROWS = "drop_invalid_rows"
    NORMALIZE_STRING = "normalize_string"
    PARSE_TIMESTAMP = "parse_timestamp"
    DEDUPLICATE = "deduplicate"
    DEFAULT_VALUE = "default_value"
    COLUMN_MAPPING = "column_mapping"


class OnError(str, Enum):
    REJECT = "reject"
    NULL = "null"
    DROP = "drop"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FixOperation(BaseModel):
    """A single declarative operation within a fix.

    Example::
        {"operation": "cast_type", "column": "amount", "from": "VARCHAR",
         "to": "DOUBLE", "on_error": "reject"}
    """

    operation: FixOperationType
    column: str | None = None
    target: str | None = None  # rename target / cast target type / etc.
    from_type: str | None = None
    to_type: str | None = None
    value: object | None = None
    on_error: OnError = OnError.REJECT


class FixProposal(BaseModel):
    """Structured proposal produced by the agent (never raw SQL)."""

    root_cause: str
    evidence: list[str] = Field(default_factory=list)
    operations: list[FixOperation] = Field(default_factory=list)
    affected_stages: list[str] = Field(default_factory=list)
    affected_columns: list[str] = Field(default_factory=list)
    expected_impact: str | None = None
    risk: RiskLevel = RiskLevel.LOW
    confidence: float = 0.0
    validation_plan: list[str] = Field(default_factory=list)
    rollback_strategy: str | None = None


class ValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationCheck(BaseModel):
    name: str
    status: ValidationStatus = ValidationStatus.PENDING
    expected: object | None = None
    observed: object | None = None
    threshold: object | None = None
    message: str | None = None


class ValidationReport(BaseModel):
    run_id: str
    checks: list[ValidationCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status == ValidationStatus.PASSED for c in self.checks)

    @property
    def failed(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == ValidationStatus.FAILED]
