"""Schema inference and registry helpers.

Maps Polars physical dtypes onto the engine-agnostic logical :class:`DataType`
enum used by the schema registry, and infers a :class:`SchemaDefinition` from a
DataFrame. The comparison engine lives in ``app/data/schema.py``.
"""

from __future__ import annotations

import polars as pl

from app.models import ColumnDefinition, DataType, SchemaDefinition

_POLAR_TO_LOGICAL: dict[str, DataType] = {
    "Int8": DataType.INTEGER,
    "Int16": DataType.INTEGER,
    "Int32": DataType.INTEGER,
    "Int64": DataType.INTEGER,
    "UInt8": DataType.INTEGER,
    "UInt16": DataType.INTEGER,
    "UInt32": DataType.INTEGER,
    "UInt64": DataType.INTEGER,
    "Float32": DataType.DOUBLE,
    "Float64": DataType.DOUBLE,
    "Utf8": DataType.VARCHAR,
    "String": DataType.VARCHAR,
    "Boolean": DataType.BOOLEAN,
    "Datetime": DataType.TIMESTAMP,
    "Date": DataType.DATE,
    "Decimal": DataType.DECIMAL,
}

_LOGICAL_TO_POLAR: dict[DataType, type[pl.DataType]] = {
    DataType.INTEGER: pl.Int64,
    DataType.DOUBLE: pl.Float64,
    DataType.VARCHAR: pl.String,
    DataType.BOOLEAN: pl.Boolean,
    DataType.TIMESTAMP: pl.Datetime,
    DataType.DATE: pl.Date,
    DataType.DECIMAL: pl.Decimal,
}


def polars_to_logical(dtype: pl.DataType) -> DataType:
    """Map a Polars dtype to a logical DataType (UNKNOWN if unrecognized)."""
    base = dtype.base_type() if hasattr(dtype, "base_type") else dtype
    name = str(base)
    return _POLAR_TO_LOGICAL.get(name, DataType.UNKNOWN)


def logical_to_polars(t: DataType) -> pl.DataType:
    """Map a logical DataType to its representative Polars dtype."""
    return _LOGICAL_TO_POLAR.get(t, pl.Unknown)


def infer_schema(
    df: pl.DataFrame, table: str, version: int = 1
) -> SchemaDefinition:
    """Infer a SchemaDefinition from a DataFrame's dtypes and nullability."""
    columns: list[ColumnDefinition] = []
    for col in df.columns:
        logical = polars_to_logical(df[col].dtype)
        columns.append(
            ColumnDefinition(
                name=col,
                type=logical,
                nullable=df[col].null_count() > 0,
            )
        )
    return SchemaDefinition(table=table, version=version, columns=columns)
