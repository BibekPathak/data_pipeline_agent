"""Agent triage state machine phase and status enums, plus the full state model."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.fix import FixProposal, ValidationReport
from app.models.quality import QualityAnomaly
from app.models.schema import DriftEvent


class Phase(str, Enum):
    """Bounded state machine phases (mirrors the spec's core agent loop)."""

    OBSERVE = "OBSERVE"
    DETECT = "DETECT"
    DIAGNOSE = "DIAGNOSE"
    PLAN = "PLAN"
    PROPOSE = "PROPOSE"
    VALIDATE = "VALIDATE"
    APPROVAL = "APPROVAL"
    STAGE = "STAGE"
    CANARY = "CANARY"
    MONITOR = "MONITOR"
    SUCCESS = "SUCCESS"
    ROLLBACK = "ROLLBACK"
    SAFE_STOP = "SAFE_STOP"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


class DeploymentStatus(str, Enum):
    NOT_DEPLOYED = "not_deployed"
    STAGED = "staged"
    CANARY_RUNNING = "canary_running"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class Hypothesis(BaseModel):
    id: str
    description: str
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    is_selected: bool = False
    fix_hint: FixProposal | None = None


class PipelineTriageState(BaseModel):
    """Explicit, typed, resumable agent state. Persisted on every transition."""

    run_id: str
    pipeline_id: str

    phase: Phase = Phase.OBSERVE

    detected_issues: list[DriftEvent] = Field(default_factory=list)
    quality_anomalies: list[QualityAnomaly] = Field(default_factory=list)

    pipeline_context: dict = Field(default_factory=dict)
    schema_context: dict = Field(default_factory=dict)
    lineage_context: dict = Field(default_factory=dict)

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    selected_hypothesis: str | None = None

    plan: list[dict] = Field(default_factory=list)

    proposed_fix: FixProposal | None = None

    validation_report: ValidationReport | None = None

    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    deployment_status: DeploymentStatus = DeploymentStatus.NOT_DEPLOYED

    canary_metrics: dict = Field(default_factory=dict)
    rollback_status: str | None = None

    final_result: str | None = None

    tool_call_count: int = 0
    iteration: int = 0
    error: str | None = None

    def bump_phase(self, next_phase: Phase) -> None:
        self.phase = next_phase
