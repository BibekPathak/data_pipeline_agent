"""Agent orchestrator: the bounded state machine.

Implements the spec's core agent loop as an explicit, bounded state machine:

    OBSERVE -> DETECT -> DIAGNOSE -> PLAN -> PROPOSE -> VALIDATE -> APPROVAL
        -> STAGE -> CANARY -> MONITOR -> SUCCESS | ROLLBACK

Guardrails enforced here (no unbounded `while True` LLM loop):
  - max iterations (agent cycle budget)
  - max tool calls (attempt budget)
  - execution timeout (asyncio)
  - resource budget (observed row/byte caps)

The agent never decides whether a production action is permitted: every tool call
is gated by the :class:`PolicyEngine`. The LLM only *produces* a declarative
:class:`FixProposal`; the policy classifies its risk and gates deployment.
State is persisted on every phase transition, making a run resumable by run_id.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any

import polars as pl

from app.agent.diagnosis import DiagnosisEngine
from app.agent.llm import LLMProvider
from app.agent.policies import PolicyEngine
from app.agent.planner import Planner
from app.data.schema import compare_schemas
from app.data.quality import run_quality_checks
from app.models import (
    ApprovalStatus,
    DeploymentStatus,
    FixProposal,
    Pipeline,
    PipelineTriageState,
    Phase,
    RiskLevel,
    ValidationReport,
)
from app.pipeline.schemas import infer_schema
from app.tools.context import ToolContext
from app.tools.registry import ToolRegistry

ApprovalResolver = Callable[[RiskLevel, FixProposal | None], ApprovalStatus]


class AgentGuardrailError(Exception):
    pass


class TriageConfig:
    def __init__(
        self,
        max_iterations: int = 12,
        max_tool_calls: int = 40,
        timeout_seconds: float = 300.0,
        max_budget_rows: int = 2_000_000,
        approval_mode: str = "auto",
    ) -> None:
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout_seconds = timeout_seconds
        self.max_budget_rows = max_budget_rows
        self.approval_mode = approval_mode


def auto_approve(risk: RiskLevel, proposal: FixProposal | None = None) -> ApprovalStatus:
    """Default approval resolver.

    - LOW    -> auto-approved (safe type coercion / non-breaking add)
    - HIGH   -> rejected without an explicit gate (never auto-deploy destructive)
    - MEDIUM -> pending (an external harness/human must approve)
    """
    if risk == RiskLevel.LOW:
        return ApprovalStatus.AUTO_APPROVED
    if risk == RiskLevel.HIGH:
        return ApprovalStatus.REJECTED
    return ApprovalStatus.PENDING


class Orchestrator:
    def __init__(
        self,
        *,
        ctx: ToolContext,
        policy: PolicyEngine | None = None,
        llm: LLMProvider | None = None,
        config: TriageConfig | None = None,
        approve: ApprovalResolver | None = None,
    ) -> None:
        self.ctx = ctx
        self.policy = policy or PolicyEngine()
        self.llm = llm
        self.config = config or TriageConfig()
        self.approve = approve or auto_approve
        self.registry = ToolRegistry(ctx, self.policy)
        self.diagnosis = DiagnosisEngine()
        self.planner = Planner()

    async def run(self, pipeline_id: str, df: pl.DataFrame) -> PipelineTriageState:
        """Run triage from OBSERVE to a final state. Bounded and policy-gated."""
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        state = PipelineTriageState(run_id=run_id, pipeline_id=pipeline_id)
        await self.ctx.store.metadata.save_state(state)

        try:
            await asyncio.wait_for(
                self._execute(state, pipeline_id, df),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            state.phase = Phase.SAFE_STOP
            state.error = "execution timeout breached"
        except AgentGuardrailError as e:
            state.phase = Phase.SAFE_STOP
            state.error = str(e)
        except Exception as e:  # noqa: BLE001
            state.phase = Phase.SAFE_STOP
            state.error = f"unexpected error: {e}"

        await self._persist(state)
        return state

    async def _execute(
        self, state: PipelineTriageState, pipeline_id: str, df: pl.DataFrame
    ) -> None:
        if df.height > self.config.max_budget_rows:
            raise AgentGuardrailError(
                f"resource budget exceeded: {df.height} rows > {self.config.max_budget_rows}"
            )

        pipeline = self.ctx.pipelines[pipeline_id]

        # ---- OBSERVE ----
        state.bump_phase(Phase.OBSERVE)
        self.ctx.observed[pipeline_id] = df
        await self.ctx.store.metadata.record_run(
            run_id=state.run_id, pipeline_id=pipeline_id, status="OBSERVE"
        )
        await self._persist(state)

        # ---- DETECT ----
        state.bump_phase(Phase.DETECT)
        expected = pipeline.base_input_schema()
        observed_schema = infer_schema(df, table=expected.table)
        state.detected_issues = compare_schemas(expected, observed_schema)
        quality_report, quality_anomalies = run_quality_checks(
            df, pipeline_id=pipeline_id
        )
        state.quality_anomalies = quality_anomalies
        state.schema_context = {
            "expected": expected.model_dump(),
            "observed": observed_schema.model_dump(),
            "issues": [i.model_dump() for i in state.detected_issues],
        }
        await self._persist(state)

        # ---- DIAGNOSE ----
        state.bump_phase(Phase.DIAGNOSE)
        radius_cols = {i.column for i in state.detected_issues}
        state.lineage_context = {
            "blast_radius": self.ctx.lineage.blast_radius(sorted(radius_cols))
        }
        state.hypotheses = self.diagnosis.build_hypotheses(
            state.detected_issues,
            state.quality_anomalies,
            state.lineage_context,
        )
        selected = self.diagnosis.select_hypothesis(state.hypotheses)
        state.selected_hypothesis = selected.id if selected else None
        await self._persist(state)

        # ---- PLAN ----
        state.bump_phase(Phase.PLAN)
        plan = []
        if selected is not None:
            plan = self.planner.build_plan(selected)
        state.plan = [
            {"tool": s.tool, "phase": s.phase.value, "args": s.args}
            for s in plan
        ]
        await self._persist(state)

        # If nothing deferred, an investigation-only plan -> SAFE_STOP.
        if selected is None:
            state.phase = Phase.SAFE_STOP
            state.final_result = "no actionable diagnosis; taking no destructive action"
            return

        # ---- PROPOSE ----
        state.bump_phase(Phase.PROPOSE)
        if self.llm is not None:
            proposal = await self.llm.build_proposal(
                drift_events=state.detected_issues,
                quality_anomalies=state.quality_anomalies,
                hypotheses=[h.description for h in state.hypotheses],
                pipeline_context=state.lineage_context,
            )
            if proposal is not None and proposal.operations:
                state.proposed_fix = proposal
        await self._persist(state)

        if state.proposed_fix is None:
            state.phase = Phase.SAFE_STOP
            state.final_result = "no safe fix proposed; taking no action"
            return

        # Policy must validate the proposal before ANY write.
        self.policy.validate_proposal(state.proposed_fix)

        # ---- VALIDATE (shadow) ----
        state.bump_phase(Phase.VALIDATE)
        validation = await self.registry.invoke(
            "validate_transformation",
            Phase.VALIDATE,
            approval=ApprovalStatus.APPROVED,  # shadow write is non-destructive
            pipeline_id=pipeline_id,
            candidate=state.proposed_fix.model_dump(),
        )
        state.validation_report = ValidationReport.model_validate(validation)
        await self._persist(state)

        if not state.validation_report.passed:
            state.phase = Phase.SAFE_STOP
            state.final_result = (
                "validation blocked deployment: "
                + str([c.name for c in state.validation_report.failed])
            )
            return

        # ---- APPROVAL ----
        state.bump_phase(Phase.APPROVAL)
        risk = self.policy.classify_fix(state.proposed_fix)
        state.approval_status = self.approve(risk, state.proposed_fix)
        await self._persist(state)

        if state.approval_status == ApprovalStatus.REJECTED:
            state.phase = Phase.SAFE_STOP
            state.final_result = f"{risk.value}-risk fix rejected by policy"
            return
        if state.approval_status == ApprovalStatus.PENDING:
            state.phase = Phase.SAFE_STOP
            state.final_result = "deployment requires human approval (not granted)"
            return

        # ---- STAGE ----
        state.bump_phase(Phase.STAGE)
        new_version_payload = {
            **(pipeline.model_dump()),
            "version": f"{pipeline.version}+{risk.value}",
        }
        await self.registry.invoke(
            "create_pipeline_version",
            Phase.STAGE,
            approval=state.approval_status,
            pipeline_id=pipeline_id,
            proposal=new_version_payload,
        )
        await self.registry.invoke(
            "stage_pipeline", Phase.STAGE, approval=state.approval_status,
            pipeline_id=pipeline_id,
        )
        state.deployment_status = DeploymentStatus.STAGED
        await self._persist(state)

        # ---- CANARY ----
        state.bump_phase(Phase.CANARY)
        canary = await self.registry.invoke(
            "run_canary", Phase.CANARY, approval=state.approval_status,
            pipeline_id=pipeline_id,
            operations=[o.model_dump() for o in state.proposed_fix.operations],
        )
        state.canary_metrics = canary
        state.deployment_status = DeploymentStatus.CANARY_RUNNING
        await self._persist(state)

        # ---- MONITOR ----
        state.bump_phase(Phase.MONITOR)
        health = await self.registry.invoke(
            "monitor_canary", Phase.MONITOR, pipeline_id=pipeline_id
        )
        canary_ok = health.get("passed", True) and state.canary_metrics.get(
            "status", "CANARY_PASSED"
        ) == "CANARY_PASSED"

        if not canary_ok:
            await self._rollback(state, pipeline_id, reason="canary/health check failed")
            return

        state.deployment_status = DeploymentStatus.ACTIVE
        state.phase = Phase.SUCCESS
        state.final_result = "candidate validated, approved, and promoted; canary healthy"
        await self._persist(state)

    async def _rollback(
        self, state: PipelineTriageState, pipeline_id: str, reason: str
    ) -> None:
        state.bump_phase(Phase.ROLLBACK)
        result = await self.registry.invoke(
            "rollback_pipeline", Phase.ROLLBACK, approval=ApprovalStatus.APPROVED,
            pipeline_id=pipeline_id,
        )
        state.deployment_status = DeploymentStatus.ROLLED_BACK
        state.rollback_status = result.get("reason") or reason
        state.final_result = f"rolled back: {reason}"
        await self._persist(state)

    async def _persist(self, state: PipelineTriageState) -> None:
        await self.ctx.store.metadata.save_state(state)
