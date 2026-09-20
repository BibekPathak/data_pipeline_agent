"""Schema definition and drift event models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DataType(str, Enum):
    """Canonical logical types used across the schema registry.

    These map onto Polars/warehouse physical types but stay engine-agnostic so
    the schema registry can grow DuckDB/Postgres adapters later.
    """

    INTEGER = "INTEGER"
    DOUBLE = "DOUBLE"
    VARCHAR = "VARCHAR"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "TIMESTAMP"
    DATE = "DATE"
    DECIMAL = "DECIMAL"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ColumnDefinition(BaseModel):
    name: str
    type: DataType
    nullable: bool = False
    description: str | None = None


class SchemaDefinition(BaseModel):
    table: str
    version: int = 1
    columns: list[ColumnDefinition] = Field(default_factory=list)

    @property
    def column_map(self) -> dict[str, ColumnDefinition]:
        return {c.name: c for c in self.columns}

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def add_version(self) -> SchemaDefinition:
        return self.model_copy(
            update={"version": self.version + 1, "columns": list(self.columns)}
        )


class DriftEventType(str, Enum):
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    TYPE_CHANGED = "type_changed"
    NULLABLE_CHANGED = "nullable_changed"
    RENAMED_COLUMN = "renamed_column"
    NESTED_STRUCTURE_CHANGED = "nested_structure_changed"


class DriftEvent(BaseModel):
    type: DriftEventType
    column: str
    expected: str | None = None
    observed: str | None = None
    severity: Severity = Severity.LOW


class RenameHypothesis(BaseModel):
    """A *possible* rename (source -> target), deliberately not treated as fact."""

    source: str
    target: str
    name_similarity: float = 0.0
    type_similarity: float = 0.0
    value_similarity: float = 0.0
    historical_support: bool = False
    confidence: float = 0.0
