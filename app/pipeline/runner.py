"""Lightweight pipeline runner.

Executes a typed :class:`Pipeline`'s stages in dependency order over a Polars
DataFrame, validating each stage's output against its declared output schema and
optionally persisting per-stage results to the warehouse under a namespace.

The runner is intentionally small and dependency-free (no Airflow/Kafka): a
pipeline is just an ordered list of stage transforms with schema boundaries.
Support for breaking into a real orchestrator later can be an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from app.data.schema import compare_schemas
from app.models import Pipeline, SchemaDefinition
from app.pipeline.schemas import infer_schema
from app.pipeline.stages import TransformationError, get_stage_transform
from app.storage.base import Warehouse


class PipelineRunError(Exception):
    """Raised when a stage fails or a schema boundary is violated."""


@dataclass
class StageResult:
    stage_id: str
    ok: bool
    input_rows: int = 0
    output_rows: int = 0
    error: str | None = None
    drift_events: list = field(default_factory=list)
    output_df: pl.DataFrame | None = None


@dataclass
class RunResult:
    pipeline_id: str
    ok: bool
    stages: list[StageResult] = field(default_factory=list)
    final_df: pl.DataFrame | None = None
    error: str | None = None

    def failed_stages(self) -> list[StageResult]:
        return [s for s in self.stages if not s.ok]

    def first_stage(self) -> StageResult | None:
        return self.stages[0] if self.stages else None


def _topological_order(pipeline: Pipeline) -> list[str]:
    """Return stage ids in dependency order (stable)."""
    ids = [s.id for s in pipeline.stages]
    deps: dict[str, set[str]] = {s.id: set(s.dependencies) for s in pipeline.stages}
    order: list[str] = []
    remaining = set(ids)
    while remaining:
        ready = [i for i in remaining if deps[i] <= set(order)]
        if not ready:
            raise PipelineRunError("Cyclic or unsatisfiable pipeline dependencies")
        picked = ready[0]
        order.append(picked)
        remaining.discard(picked)
    return order


def _validate_against_schema(
    df: pl.DataFrame, schema: SchemaDefinition, table: str
) -> list:
    """Return drift events (failures) if df doesn't match the declared schema."""
    inferred = infer_schema(df, table=table)
    return compare_schemas(schema, inferred)


async def run_pipeline(
    pipeline: Pipeline,
    df: pl.DataFrame,
    warehouse: Warehouse | None = None,
    *,
    namespace: str = "main",
    persist: bool = True,
    strict_schema: bool = False,
) -> RunResult:
    """Run a pipeline over ``df``.

    When ``warehouse`` is given and ``persist`` is True, the result of each stage
    is written to ``<namespace>_<table>`` in the warehouse, enabling shadow runs
    (namespace="shadow") vs production (namespace="main").
    """
    results: list[StageResult] = []
    current = df
    error: str | None = None
    order = _topological_order(pipeline)
    stage_by_id = pipeline.stage_map()

    for stage_id in order:
        stage = stage_by_id[stage_id]
        try:
            transform = get_stage_transform(stage.transformation)
            output = transform(current, stage.config)
            events = _validate_against_schema(output, stage.output_schema, stage.name)
            if events and strict_schema:
                raise PipelineRunError(
                    f"Stage {stage_id} output schema mismatch: {[e.type.value for e in events]}"
                )
            results.append(
                StageResult(
                    stage_id=stage_id,
                    ok=True,
                    input_rows=current.height,
                    output_rows=output.height,
                    drift_events=events,
                    output_df=output,
                )
            )
            if warehouse is not None and persist:
                await warehouse.write_table(
                    f"{stage.name}", f"{namespace}:v{pipeline.version}", output
                )
            current = output
        except (TransformationError, PipelineRunError) as exc:
            results.append(
                StageResult(
                    stage_id=stage_id,
                    ok=False,
                    input_rows=current.height,
                    error=str(exc),
                )
            )
            error = str(exc)
            break

    return RunResult(
        pipeline_id=pipeline.id, ok=error is None, stages=results,
        final_df=current if error is None else None, error=error,
    )
