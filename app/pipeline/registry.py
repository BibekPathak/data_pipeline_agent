"""Pipeline registry: load typed Pipeline definitions from disk and metadata.

Pipelines ship as JSON files (or in the metadata store) and are materialized
into :class:`Pipeline` models. Stage schema boundaries are validated lazily by
the runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.models import Pipeline


class PipelineRegistry:
    def __init__(self, directory: str | Path | None = None) -> None:
        self._directory = Path(directory) if directory else None
        self._overrides: dict[str, Pipeline] = {}

    def register(self, pipeline: Pipeline) -> None:
        self._overrides[pipeline.id] = pipeline

    def _load_from_disk(self) -> list[Pipeline]:
        if self._directory is None or not self._directory.exists():
            return []
        pipelines: list[Pipeline] = []
        for path in sorted(self._directory.glob("*.json")):
            data = json.loads(path.read_text())
            pipeline = Pipeline.model_validate(data)
            pipelines.append(pipeline)
        return pipelines

    def list(self) -> list[Pipeline]:
        merged: dict[str, Pipeline] = {p.id: p for p in self._load_from_disk()}
        merged.update(self._overrides)
        return list(merged.values())

    def get(self, pipeline_id: str) -> Pipeline | None:
        return {p.id: p for p in self.list()}.get(pipeline_id)

    def __len__(self) -> int:
        return len(self.list())
