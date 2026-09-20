"""Pipeline tools: pipeline/run/stage introspection and shadow stage runs."""

from __future__ import annotations

from app.agent.policies import ActionClass
from app.data.lineage import LineageEdge
from app.tools.base import tool
from app.tools.context import ToolContext


class PipelineTools:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def tools(self) -> list:
        return [
            self.get_pipeline,
            self.get_pipeline_run,
            self.get_stage_logs,
            self.get_lineage,
            self.run_stage,
        ]

    @tool("get_pipeline", ActionClass.READ_ONLY, "Return the pipeline definition.")
    async def get_pipeline(self, pipeline_id: str) -> dict:
        p = self.ctx.pipelines[pipeline_id]
        return p.model_dump()

    @tool("get_pipeline_run", ActionClass.READ_ONLY, "Return the latest run record for a pipeline.")
    async def get_pipeline_run(self, pipeline_id: str) -> dict:
        run = await self.ctx.store.metadata.get_run(pipeline_id)
        return run or {"pipeline_id": pipeline_id, "status": "no_record"}

    @tool("get_stage_logs", ActionClass.READ_ONLY, "Return persisted stage logs for a pipeline run.")
    async def get_stage_logs(self, pipeline_id: str) -> dict:
        run = await self.ctx.store.metadata.get_run(pipeline_id)
        return {"pipeline_id": pipeline_id, "stages": run or {}}

    @tool("get_lineage", ActionClass.READ_ONLY, "Return the lineage graph and blast radius for changed columns.")
    async def get_lineage(self, changed_columns: list[str] | None = None) -> dict:
        nodes = self.ctx.lineage.nodes
        edges = [
            {"source": e.source, "destination": e.destination, "columns": e.columns}
            for e in self.ctx.lineage.all_edges
        ]
        radius = (
            self.ctx.lineage.blast_radius(changed_columns) if changed_columns else {}
        )
        return {"nodes": nodes, "edges": edges, "blast_radius": radius}

    @tool("run_stage", ActionClass.SHADOW_WRITE, "Run a stage transformation in shadow namespace.")
    async def run_stage(self, pipeline_id: str, stage_id: str) -> dict:
        p = self.ctx.pipelines[pipeline_id]
        stage = p.stage_map()[stage_id]
        df = self.ctx.observed[pipeline_id]
        from app.pipeline.stages import get_stage_transform

        transform = get_stage_transform(stage.transformation)
        output = transform(df, stage.config)
        await self.ctx.store.warehouse.write_table(
            stage.name, f"{self.ctx.shadow_namespace}:probe", output
        )
        return {"stage_id": stage_id, "output_rows": output.height, "namespace": self.ctx.shadow_namespace}
