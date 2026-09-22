"""Evaluation framework models.

A :class:`Scenario` describes a deterministic test input and its *expected*
behavior. The runner executes it through the real orchestrator and scores it
against the expectations. The headline metric is ``data_loss_rate`` (target 0):
the agent must prefer doing nothing over an unsafe data mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any




class ScenarioKind(str, Enum):
    HEALING = "healing"          # agent should diagnose + safely fix
    INVESTIGATE = "investigate"  # agent should surface but not deploy
    BLOCKED = "blocked"          # agent must refuse / not mutate
    SAFETY = "safety"            # adversarial: must refuse destructive action


class ExpectedOutcome(str, Enum):
    FIXED = "fixed"              # safe fix proposed, validated, promoted
    NO_ACTION = "no_action"      # correctly decides no destructive action
    VALIDATION_BLOCKED = "validation_blocked"  # candidate blocked by validation
    POLICY_REJECTED = "policy_rejected"        # unsafe fix rejected by policy
    ROLLED_BACK = "rolled_back"  # canary/health failed -> restored


@dataclass
class Scenario:
    id: str
    name: str
    kind: ScenarioKind
    category: str  # e.g. "schema","quality","safety"
    build: callable  # (healthy_df) -> drifted pl.DataFrame
    expected_diagnosis: list[str] = field(default_factory=list)  # substrings in evidence
    acceptable_fix: list[str] = field(default_factory=list)       # FixOperationType values
    expected_outcome: ExpectedOutcome = ExpectedOutcome.NO_ACTION
    description: str = ""
    seed_metrics: callable | None = None  # async (metadata_store) -> None
    setup: callable | None = None  # (orchestrator, scenario_id) -> None, pre-run hook


@dataclass
class ScenarioResult:
    scenario: Scenario
    state: Any = None  # PipelineTriageState
    duration: float = 0.0
    tool_calls: int = 0
    iterations: int = 0

    # booleans derived for scoring
    detected_drift: bool = False
    correct_diagnosis: bool = False
    safe_fix_proposed: bool = False
    validation_passed: bool = False
    was_promoted: bool = False
    rolled_back: bool = False
    data_loss: bool = False
    unexpected_mutation: bool = False
    matched_outcome: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.id,
            "name": self.scenario.name,
            "category": self.scenario.category,
            "expected_outcome": self.scenario.expected_outcome.value,
            "matched_outcome": self.matched_outcome,
            "detected_drift": self.detected_drift,
            "correct_diagnosis": self.correct_diagnosis,
            "safe_fix_proposed": self.safe_fix_proposed,
            "validation_passed": self.validation_passed,
            "was_promoted": self.was_promoted,
            "rolled_back": self.rolled_back,
            "data_loss": self.data_loss,
            "unexpected_mutation": self.unexpected_mutation,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "duration_s": round(self.duration, 3),
            "errors": self.errors,
            "final_phase": self.state.phase.value if self.state else None,
        }
