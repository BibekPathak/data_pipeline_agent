"""Pipeline stage transformations.

Contains the *constrained* set of safe transformations the engine is allowed to
apply. These same operations back both named pipeline stages (via the
``transformation`` registry) and declarative fix proposals. Arbitrary SQL is
deliberately not supported: each operation is a small, reviewed Polars
transformation.
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from app.models import DataType, FixOperation, FixOperationType, OnError
from app.pipeline.schemas import logical_to_polars, polars_to_logical


class TransformationError(Exception):
    """Raised when a transformation cannot be applied safely."""


# --- Primitive operations (used by fix proposals) ---


def op_cast_type(
    df: pl.DataFrame, column: str, to: str, on_error: OnError = OnError.REJECT
) -> pl.DataFrame:
    try:
        data_type = DataType(to)
    except ValueError as exc:
        raise TransformationError(f"Unknown target type {to!r}") from exc
    target = logical_to_polars(data_type)
    series = df[column]

    if on_error == OnError.REJECT:
        try:
            cast = series.cast(target, strict=True)
        except Exception as exc:
            raise TransformationError(
                f"Cannot cast {column} to {to}: {exc}"
            ) from exc
        return df.with_columns(cast.alias(column))

    # NULL / DROP: non-strict cast turns invalid values into nulls
    cast = series.cast(target, strict=False)
    if on_error == OnError.DROP:
        return df.with_columns(cast.alias(column)).filter(
            pl.col(column).is_not_null()
        )
    return df.with_columns(cast.alias(column))


def op_fill_null(df: pl.DataFrame, column: str, value: Any) -> pl.DataFrame:
    return df.with_columns(df[column].fill_null(value).alias(column))


def op_deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique()


def op_normalize_string(
    df: pl.DataFrame, column: str, *, lower: bool = False, strip: bool = True
) -> pl.DataFrame:
    expr = pl.col(column)
    if strip:
        expr = expr.str.strip_chars()
    if lower:
        expr = expr.str.to_lowercase()
    return df.with_columns(expr.alias(column))


def op_parse_timestamp(df: pl.DataFrame, column: str, fmt: str | None = None) -> pl.DataFrame:
    try:
        if fmt:
            parsed = df[column].str.strptime(pl.Datetime, fmt)
        else:
            parsed = df[column].str.to_datetime()
    except Exception as exc:
        raise TransformationError(f"Cannot parse {column} as timestamp: {exc}") from exc
    return df.with_columns(parsed.alias(column))


def op_rename_column(df: pl.DataFrame, column: str, target: str) -> pl.DataFrame:
    if target not in df.columns:
        return df.rename({column: target})
    raise TransformationError(f"Cannot rename {column} -> {target}: target exists")


def op_default_value(
    df: pl.DataFrame, column: str, value: Any, condition: str | None = None
) -> pl.DataFrame:
    new_col = df[column].fill_null(value)
    # Also replace empty strings for string columns
    if df[column].dtype in (pl.String, pl.Utf8):
        new_col = new_col.cast(pl.String).replace("", str(value))
    return df.with_columns(new_col.alias(column))


def op_drop_invalid_rows(
    df: pl.DataFrame, column: str, invalid_values: list[Any] | None = None
) -> pl.DataFrame:
    expr = pl.col(column).is_null()
    if invalid_values:
        expr = expr | pl.col(column).is_in(invalid_values)
    return df.filter(~expr)


def _coerce(value: Any, dtype: pl.DataType) -> Any:
    """Coerce a python scalar to be dtype-compatible for replace operations."""
    try:
        if dtype.is_float():
            return float(value)
        if dtype.is_integer():
            return int(float(value))
        return str(value)
    except (TypeError, ValueError):
        return value


def op_column_mapping(
    df: pl.DataFrame, column: str, mapping: dict[str, Any], default: Any = None
) -> pl.DataFrame:
    # Coerce keys/values/default to the column dtype so a declared mapping
    # applies uniformly. A corrupting mapping must EXECUTE and be caught by
    # validation, not crash the engine.
    dtype = df[column].dtype
    old = [_coerce(v, dtype) for v in mapping.keys()]
    new = [_coerce(v, dtype) for v in mapping.values()]
    if default is not None:
        mapped = df[column].replace_strict(old, new, default=_coerce(default, dtype))
    else:
        mapped = df[column].replace(old, new)
    return df.with_columns(mapped.alias(column))


# --- Fix operation dispatch ---

_FIX_HANDLERS: dict[FixOperationType, Callable[..., pl.DataFrame]] = {
    FixOperationType.CAST_TYPE: op_cast_type,
    FixOperationType.FILL_NULL: lambda df, column, value, **_: op_fill_null(df, column, value),
    FixOperationType.DEDUPLICATE: lambda df, **_: op_deduplicate(df),
    FixOperationType.NORMALIZE_STRING: lambda df, column, lower=False, strip=True, **_: op_normalize_string(df, column, lower=lower, strip=strip),
    FixOperationType.PARSE_TIMESTAMP: lambda df, column, fmt=None, **_: op_parse_timestamp(df, column, fmt),
    FixOperationType.RENAME_COLUMN: lambda df, column, target, **_: op_rename_column(df, column, target),
    FixOperationType.DEFAULT_VALUE: lambda df, column, value, **_: op_default_value(df, column, value),
    FixOperationType.DROP_INVALID_ROWS: lambda df, column, invalid_values=None, **_: op_drop_invalid_rows(df, column, invalid_values),
    FixOperationType.COLUMN_MAPPING: lambda df, column, mapping, default=None, **_: op_column_mapping(df, column, mapping, default),
}


def apply_fix_operations(df: pl.DataFrame, ops: list[FixOperation]) -> pl.DataFrame:
    """Apply a sequence of declarative fix operations in order.

    Raises :class:`TransformationError` on any unsafe/un-applicable operation.
    Only passes keyword arguments that the specific operation type accepts.
    """
    result = df
    for op in ops:
        handler = _FIX_HANDLERS[op.operation]
        kwargs = _build_kwargs(op)
        result = handler(result, **kwargs)
    return result


def _build_kwargs(op: FixOperation) -> dict:
    """Build the kwargs an operation handler expects from a FixOperation."""
    kwargs: dict[str, object] = {}
    if op.column is not None:
        kwargs["column"] = op.column

    if op.operation == FixOperationType.CAST_TYPE:
        if op.to_type is not None:
            kwargs["to"] = op.to_type
        if op.on_error is not None:
            kwargs["on_error"] = op.on_error
    elif op.operation in (
        FixOperationType.FILL_NULL,
        FixOperationType.DEFAULT_VALUE,
    ):
        if op.value is not None:
            kwargs["value"] = op.value
    elif op.operation == FixOperationType.RENAME_COLUMN:
        if op.target is not None:
            kwargs["target"] = op.target
    elif op.operation == FixOperationType.NORMALIZE_STRING:
        opts = set((op.target or "").split(","))
        kwargs["lower"] = "lower" in opts
        kwargs["strip"] = "nostrip" not in opts
    elif op.operation == FixOperationType.PARSE_TIMESTAMP:
        if op.target is not None:
            kwargs["fmt"] = op.target
    elif op.operation == FixOperationType.DROP_INVALID_ROWS:
        if op.value is not None:
            kwargs["invalid_values"] = op.value
    elif op.operation == FixOperationType.COLUMN_MAPPING:
        if op.value is not None:
            kwargs["mapping"] = op.value
        if op.target is not None:
            kwargs["default"] = op.target

    return kwargs


# --- Named stage transformations (registry) ---

_STAGE_TRANSFORMS: dict[str, Callable[[pl.DataFrame, dict], pl.DataFrame]] = {}


def register(name: str) -> Callable:
    def deco(fn: Callable[[pl.DataFrame, dict], pl.DataFrame]) -> Callable:
        _STAGE_TRANSFORMS[name] = fn
        return fn

    return deco


@register("clean_orders")
def _clean_orders(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    return df.filter(pl.col("order_id").is_not_null())


@register("enrich_orders")
def _enrich_orders(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    if "amount" in df.columns and df["amount"].dtype.is_numeric():
        return df.with_columns((pl.col("amount") * 1.0).alias("amount"))
    return df


@register("aggregate_daily_revenue")
def _aggregate_daily_revenue(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    agg_cols: list[pl.Expr] = []
    if "currency" in df.columns:
        agg_cols.append(pl.col("currency").first())
    if "amount" in df.columns and df["amount"].dtype.is_numeric():
        agg_cols.append(pl.sum("amount").alias("revenue"))
    if "created_at" in df.columns:
        agg_cols.append(pl.min("created_at").alias("earliest_created_at"))
    if not agg_cols:
        return df
    return df.group_by(["order_id"]).agg(agg_cols)


def get_stage_transform(name: str) -> Callable[[pl.DataFrame, dict], pl.DataFrame]:
    if name not in _STAGE_TRANSFORMS:
        raise TransformationError(f"Unknown stage transformation {name!r}")
    return _STAGE_TRANSFORMS[name]


def list_stage_transforms() -> list[str]:
    return sorted(_STAGE_TRANSFORMS)
