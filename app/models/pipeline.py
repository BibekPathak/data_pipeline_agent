"""Pipeline and pipeline stage definitions (typed, declarative)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.schema import SchemaDefinition


class PipelineStage(BaseModel):
    id: str
    name: str
    input_schema: SchemaDefinition
    output_schema: SchemaDefinition
    transformation: str  # identifier into the transformation registry
    config: dict = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


class Pipeline(BaseModel):
    id: str
    name: str
    version: str
    source: str  # logical source key, e.g. "orders" / "fixtures:orders.csv"
    stages: list[PipelineStage] = Field(default_factory=list)
    schedule: str | None = None

    def stage_map(self) -> dict[str, PipelineStage]:
        return {s.id: s for s in self.stages}

    def base_input_schema(self) -> SchemaDefinition:
        """Schema the first stage expects; used as the baseline for drift checks."""
        first = self.stages[0]
        return first.input_schema
