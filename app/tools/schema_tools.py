"""Schema tools: expected/observed schema, comparison, and history."""

from __future__ import annotations

from app.agent.policies import ActionClass
from app.data.schema import compare_schemas, compute_rename_hypotheses
from app.pipeline.schemas import infer_schema
from app.tools.base import tool
from app.tools.context import ToolContext


class SchemaTools:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def tools(self) -> list:
        return [self.get_expected_schema, self.get_observed_schema, self.compare_schemas, self.get_schema_history]

    @tool("get_expected_schema", ActionClass.READ_ONLY, "Return the pipeline's expected input schema.")
    async def get_expected_schema(self, pipeline_id: str) -> dict:
        pipeline = self.ctx.pipelines[pipeline_id]
        return pipeline.base_input_schema().model_dump()

    @tool("get_observed_schema", ActionClass.READ_ONLY, "Return the inferred schema of the latest observed data.")
    async def get_observed_schema(self, pipeline_id: str) -> dict:
        df = self.ctx.observed[pipeline_id]
        observed = infer_schema(df, table=df.columns[0] if df.columns else "data")
        return observed.model_dump()

    @tool("compare_schemas", ActionClass.READ_ONLY, "Compare expected vs observed schemas, return drift events.")
    async def compare_schemas(self, pipeline_id: str, include_rename: bool = True) -> dict:
        pipeline = self.ctx.pipelines[pipeline_id]
        expected = pipeline.base_input_schema()
        df = self.ctx.observed[pipeline_id]
        observed = infer_schema(df, table=expected.table)
        events = [e.model_dump() for e in compare_schemas(expected, observed)]
        hypotheses = []
        if include_rename:
            hypotheses = [
                h.model_dump()
                for h in compute_rename_hypotheses(expected, observed, df)
            ]
        return {"drift_events": events, "rename_hypotheses": hypotheses}

    @tool("get_schema_history", ActionClass.READ_ONLY, "Return historical schema versions for a table.")
    async def get_schema_history(self, table: str) -> dict:
        versions = await self.ctx.store.metadata.list_schema_versions(table)
        schemas = []
        for v in versions:
            s = await self.ctx.store.metadata.get_schema(table, v)
            if s is not None:
                schemas.append(s.model_dump())
        return {"table": table, "versions": schemas}
