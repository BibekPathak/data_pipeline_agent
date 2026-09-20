"""Shared context injected into tool handlers."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from app.data.lineage import LineageGraph
from app.models import Pipeline
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
