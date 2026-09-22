"""Shared context injected into tool handlers."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from app.data.lineage import LineageGraph
from app.models import Pipeline
from app.rollback.manager import RollbackManager
from app.storage.base import BackendStore


@dataclass
class ToolContext:
    store: BackendStore
    pipelines: dict[str, Pipeline] = field(default_factory=dict)
    lineage: LineageGraph = field(default_factory=LineageGraph)
    # Mapping from pipeline id -> observed DataFrame (the latest ingested input).
    observed: dict[str, pl.DataFrame] = field(default_factory=dict)
    # Active pipeline version namespace in the warehouse (e.g. "main").
    active_namespace: str = "main"
    shadow_namespace: str = "shadow"
    # Reference tables for referential-integrity checks: {fk_column: reference_df}
    references: dict[str, pl.DataFrame] = field(default_factory=dict)
    # Shared, deterministic version-restore primitive used by deployment tools.
    rollback: RollbackManager | None = None


def make_context(store: BackendStore, **kwargs) -> ToolContext:
    """Build a ToolContext with a default RollbackManager bound to the store."""
    ctx = ToolContext(store=store, **kwargs)
    if ctx.rollback is None:
        ctx.rollback = RollbackManager(store.warehouse, store.metadata)
    return ctx
