"""Tests for the policy engine and safety classification."""

from __future__ import annotations

import pytest

from app.agent.policies import ActionClass, PolicyEngine, PolicyError
from app.models import (
    ApprovalStatus,
    FixOperation,
    FixOperationType,
    FixProposal,
    RiskLevel,
)
from app.models.state import Phase


def _proposal(risk: RiskLevel, *ops: FixOperation) -> FixProposal:
    return FixProposal(root_cause="test", operations=list(ops), risk=risk)


def _cast() -> FixOperation:
    return FixOperation(operation=FixOperationType.CAST_TYPE, column="a", to_type="DOUBLE")


class TestSafetyClassification:
    def test_tool_action_classes(self):
        p = PolicyEngine()
        assert p.classify_tool("profile_dataset") == ActionClass.READ_ONLY
        assert p.classify_tool("run_quality_checks") == ActionClass.READ_ONLY
        assert p.classify_tool("validate_transformation") == ActionClass.SHADOW_WRITE
        assert p.classify_tool("stage_pipeline") == ActionClass.STAGING_WRITE
        assert p.classify_tool("run_canary") == ActionClass.PRODUCTION_WRITE
        assert p.classify_tool("rollback_pipeline") == ActionClass.ROLLBACK

    def test_unknown_tool_rejected(self):
        with pytest.raises(PolicyError):
            PolicyEngine().classify_tool("drop_table")

    def test_phase_gating(self):
        p = PolicyEngine()
        # READ_ONLY allowed in OBSERVE, PRODUCTION_WRITE not.
        assert p.tool_allowed_in_phase("profile_dataset", Phase.OBSERVE)
        assert not p.tool_allowed_in_phase("run_canary", Phase.OBSERVE)
        # STAGING allowed in STAGE
        assert p.tool_allowed_in_phase("stage_pipeline", Phase.STAGE)
        # ROLLBACK only in ROLLBACK phase
        assert p.tool_allowed_in_phase("rollback_pipeline", Phase.ROLLBACK)
        assert not p.tool_allowed_in_phase("rollback_pipeline", Phase.OBSERVE)


class TestRiskClassification:
    def test_cast_is_low(self):
        p = PolicyEngine()
        assert p.classify_fix(_proposal(RiskLevel.LOW, _cast())) == RiskLevel.LOW

    def test_drop_invalid_is_high(self):
        p = PolicyEngine()
        op = FixOperation(operation=FixOperationType.DROP_INVALID_ROWS, column="a")
        assert p.classify_fix(_proposal(RiskLevel.LOW, op)) == RiskLevel.HIGH

    def test_rename_is_medium(self):
        p = PolicyEngine()
        op = FixOperation(operation=FixOperationType.RENAME_COLUMN, column="a", target="b")
        assert p.classify_fix(_proposal(RiskLevel.LOW, op)) == RiskLevel.MEDIUM

    def test_proposal_validated(self):
        p = PolicyEngine()
        # A proposal with an operation outside the enum can't even be built via
        # the model; validate guards against a missing to_type.
        with pytest.raises(PolicyError):
            p.validate_proposal(
                _proposal(
                    RiskLevel.LOW,
                    FixOperation(operation=FixOperationType.CAST_TYPE, column="a"),
                )
            )


class TestApproval:
    def test_auto_low_approves(self):
        p = PolicyEngine(approval_mode="auto")
        ok, _ = p.can_deploy(_proposal(RiskLevel.LOW, _cast()), ApprovalStatus.NOT_REQUIRED)
        assert ok is True

    def test_auto_high_rejected(self):
        p = PolicyEngine(approval_mode="auto")
        ok, _ = p.can_deploy(
            _proposal(RiskLevel.HIGH, FixOperation(operation=FixOperationType.DROP_INVALID_ROWS, column="a")),
            ApprovalStatus.NOT_REQUIRED,
        )
        assert ok is False

    def test_medium_requires_approval(self):
        p = PolicyEngine(approval_mode="auto")
        op = FixOperation(operation=FixOperationType.RENAME_COLUMN, column="a", target="b")
        ok, _ = p.can_deploy(_proposal(RiskLevel.MEDIUM, op), ApprovalStatus.PENDING)
        assert ok is False
        ok2, _ = p.can_deploy(_proposal(RiskLevel.MEDIUM, op), ApprovalStatus.APPROVED)
        assert ok2 is True

    def test_production_write_gated_by_approval(self):
        p = PolicyEngine()
        with pytest.raises(PolicyError):
            p.assert_action_allowed("run_canary", Phase.CANARY, ApprovalStatus.REJECTED)
