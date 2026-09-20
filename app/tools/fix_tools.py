"""Fix tools: proposal, shadow validation, and pipeline version creation.

Shadow validation is the most important safety primitive: a candidate fix is run
on the *observed* data in a shadow namespace and compared against the current
output on row count / schema / null rate / aggregates / business invariants.
A candidate that destroys data or silently corrupts business metrics is blocked.
"""

from __future__ import annotations

import polars as pl

from app.agent.policies import ActionClass
from app.models import FixOperation, Pipeline, ValidationReport
from app.pipeline.runner import run_pipeline
from app.pipeline.stages import apply_fix_operations
from app.tools.base import tool
from app.tools.context import ToolContext
from app.validation.engine import shadow_validate


class FixTools:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def tools(self) -> list:
        return [
            self.propose_transformation,
            self.validate_transformation,
            self.create_pipeline_version,
        ]

    @tool("propose_transformation", ActionClass.READ_ONLY,
          "Build a structured fix proposal from diagnostic evidence.")
    async def propose_transformation(self, pipeline_id: str, llm) -> dict:
        # Evidence is gathered by the orchestrator and passed via llm.build_proposal.
        # This tool is informational; the orchestrator holds the proposal in state.
        return {"status": "handled_by_orchestrator", "pipeline_id": pipeline_id}

    @tool("validate_transformation", ActionClass.SHADOW_WRITE,
          "Shadow-run the candidate pipeline and produce a validation report.")
    async def validate_transformation(self, pipeline_id: str, candidate: dict) -> dict:
        ops = [FixOperation.model_validate(o) for o in candidate.get("operations", [])]
        pipeline = self.ctx.pipelines[pipeline_id]
        df = self.ctx.observed[pipeline_id]

        corrected = apply_fix_operations(df, ops)
        candidate_pipe = pipeline.model_copy(update={"version": f"{pipeline.version}+fix"})
        expected_schema = (
            candidate_pipe.stages[-1].output_schema
            if candidate_pipe.stages else None
        )

        current_run = await run_pipeline(
            pipeline, df, self.ctx.store.warehouse,
            namespace=self.ctx.active_namespace, persist=False,
        )
        candidate_run = await run_pipeline(
            candidate_pipe, corrected, self.ctx.store.warehouse,
            namespace=self.ctx.shadow_namespace, persist=False,
        )
        report = shadow_validate(
            current_run, candidate_run,
            pipeline_id=pipeline_id,
            expected_schema=expected_schema,
            input_row_count=df.height,
        )
        return report.model_dump()

    @tool("create_pipeline_version", ActionClass.STAGING_WRITE,
          "Persist a new pipeline version with the fix applied.")
    async def create_pipeline_version(self, pipeline_id: str, proposal: dict) -> dict:
        pipeline = self.ctx.pipelines[pipeline_id]
        new_version = Pipeline.model_validate(proposal)
        new_version.id = pipeline_id
        self.ctx.pipelines[pipeline_id] = new_version
        await self.ctx.store.metadata.save_pipeline(new_version)
        return {"pipeline_id": pipeline_id, "new_version": new_version.version}
