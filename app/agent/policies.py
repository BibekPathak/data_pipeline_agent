"""Action policies and safety classification.

The policy engine decides whether any action is permitted and whether a fix may
be staged/deployed. The LLM (or any agent reasoning) never makes this decision
directly: it only produces a declarative :class:`FixProposal`, and the policy
engine classifies it and gates execution.

Risk-based approval rules (per spec):
  - LOW    : safe type coercion / non-breaking column addition -> auto-stage
  - MEDIUM : row filtering / column rename -> require approval
  - HIGH   : data deletion / large row-count change / prod schema mutation
             -> require approval + explicit confirmation
"""

from __future__ import annotations

from enum import Enum

from app.models import ApprovalStatus, FixOperation, FixOperationType, FixProposal, RiskLevel
from app.models.state import Phase


class ActionClass(str, Enum):
    READ_ONLY = "READ_ONLY"
    SHADOW_WRITE = "SHADOW_WRITE"
    STAGING_WRITE = "STAGING_WRITE"
    PRODUCTION_WRITE = "PRODUCTION_WRITE"
    ROLLBACK = "ROLLBACK"


# Tools the agent may call, mapped to their safety class. This mapping is the
# single source of truth for "can this tool touch production?"
TOOL_ACTION_CLASS: dict[str, ActionClass] = {
    # schema
    "get_expected_schema": ActionClass.READ_ONLY,
    "get_observed_schema": ActionClass.READ_ONLY,
    "compare_schemas": ActionClass.READ_ONLY,
    "get_schema_history": ActionClass.READ_ONLY,
    # data
    "profile_dataset": ActionClass.READ_ONLY,
    "query_dataset": ActionClass.READ_ONLY,
    "sample_rows": ActionClass.READ_ONLY,
    "run_quality_checks": ActionClass.READ_ONLY,
    "compare_distributions": ActionClass.READ_ONLY,
    # pipeline
    "get_pipeline": ActionClass.READ_ONLY,
    "get_pipeline_run": ActionClass.READ_ONLY,
    "get_stage_logs": ActionClass.READ_ONLY,
    "get_lineage": ActionClass.READ_ONLY,
    "run_stage": ActionClass.SHADOW_WRITE,
    # fix
    "propose_transformation": ActionClass.READ_ONLY,
    "validate_transformation": ActionClass.SHADOW_WRITE,
    "create_pipeline_version": ActionClass.STAGING_WRITE,
    # deployment
    "stage_pipeline": ActionClass.STAGING_WRITE,
    "run_canary": ActionClass.PRODUCTION_WRITE,
    "monitor_canary": ActionClass.READ_ONLY,
    "rollback_pipeline": ActionClass.ROLLBACK,
}

# Which phases are allowed to invoke a given action class.
_PHASE_ACTION_ALLOW: dict[Phase, set[ActionClass]] = {
    Phase.OBSERVE: {ActionClass.READ_ONLY},
    Phase.DETECT: {ActionClass.READ_ONLY},
    Phase.DIAGNOSE: {ActionClass.READ_ONLY},
    Phase.PLAN: {ActionClass.READ_ONLY},
    Phase.PROPOSE: {ActionClass.READ_ONLY},
    Phase.VALIDATE: {ActionClass.SHADOW_WRITE, ActionClass.READ_ONLY},
    Phase.APPROVAL: {ActionClass.READ_ONLY},
    Phase.STAGE: {ActionClass.STAGING_WRITE, ActionClass.READ_ONLY},
    Phase.CANARY: {ActionClass.PRODUCTION_WRITE, ActionClass.READ_ONLY},
    Phase.MONITOR: {ActionClass.READ_ONLY},
    Phase.ROLLBACK: {ActionClass.ROLLBACK, ActionClass.READ_ONLY},
    Phase.SUCCESS: set(),
    Phase.SAFE_STOP: set(),
}


class PolicyError(Exception):
    """Raised when a tool call or fix is not permitted."""


class PolicyEngine:
    def __init__(self, approval_mode: str = "auto") -> None:
        self.approval_mode = approval_mode

    def classify_tool(self, tool_name: str) -> ActionClass:
        if tool_name not in TOOL_ACTION_CLASS:
            raise PolicyError(f"Unknown tool {tool_name!r} (not in tool registry)")
        return TOOL_ACTION_CLASS[tool_name]

    def tool_allowed_in_phase(self, tool_name: str, phase: Phase) -> bool:
        cls = self.classify_tool(tool_name)
        return cls in _PHASE_ACTION_ALLOW.get(phase, set())

    def classify_fix(self, proposal: FixProposal) -> RiskLevel:
        """Derive a risk level from a proposal (never from the LLM directly)."""
        if proposal.risk == RiskLevel.HIGH:
            return RiskLevel.HIGH
        destructive = any(
            o.operation in (FixOperationType.DROP_INVALID_ROWS,)
            for o in proposal.operations
        )
        renames = any(o.operation == FixOperationType.RENAME_COLUMN for o in proposal.operations)
        if destructive:
            return RiskLevel.HIGH
        if renames or any(o.operation == FixOperationType.DEDUPLICATE for o in proposal.operations):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def required_approval(self, risk: RiskLevel) -> ApprovalStatus | None:
        """Return the approval status required for a given risk.

        ``None`` means no approval gate (auto). Returns PENDING when manual
        approval is required for this risk in the current mode.
        """
        if risk == RiskLevel.LOW:
            return ApprovalStatus.AUTO_APPROVED
        if self.approval_mode == "auto":
            # In auto mode we still gate MEDIUM/HIGH behind an explicit decision
            # by the approving harness; unsafe/highly-destructive are rejected.
            if risk == RiskLevel.HIGH:
                return ApprovalStatus.REJECTED
            return ApprovalStatus.PENDING
        # manual / human_in_loop
        return ApprovalStatus.PENDING

    def validate_proposal(self, proposal: FixProposal) -> None:
        """Reject proposals containing anything outside the constrained set."""
        allowed = {op for op in FixOperationType}
        for op in proposal.operations:
            if op.operation not in allowed:
                raise PolicyError(
                    f"Operation {op.operation} is not in the constrained allow-list"
                )
            if op.operation == FixOperationType.CAST_TYPE and not op.to_type:
                raise PolicyError("cast_type requires a to_type")

    def can_deploy(
        self, proposal: FixProposal, approval: ApprovalStatus
    ) -> tuple[bool, str]:
        """Whether a proposal may be staged/deployed given its approval status."""
        risk = self.classify_fix(proposal)
        required = self.required_approval(risk)
        if required == ApprovalStatus.REJECTED:
            return False, "HIGH-risk fix rejected without explicit gate"
        if required == ApprovalStatus.PENDING:
            return approval in (
                ApprovalStatus.APPROVED,
                ApprovalStatus.AUTO_APPROVED,
            ), f"requires approval, got {approval.value}"
        if required == ApprovalStatus.AUTO_APPROVED:
            return True, "LOW-risk fix auto-approved"
        return False, "no approval pathway"

    def assert_action_allowed(
        self, tool_name: str, phase: Phase, approval: ApprovalStatus | None = None
    ) -> None:
        """Raise PolicyError if the tool is not permitted in the current phase."""
        cls = self.classify_tool(tool_name)
        if cls not in _PHASE_ACTION_ALLOW.get(phase, set()):
            raise PolicyError(
                f"Tool {tool_name!r} ({cls.value}) not permitted in phase {phase.value}"
            )
        if cls in (ActionClass.PRODUCTION_WRITE, ActionClass.STAGING_WRITE):
            if approval in (None, ApprovalStatus.REJECTED):
                raise PolicyError(
                    f"{tool_name!r} blocked: no production/staging approval"
                )
