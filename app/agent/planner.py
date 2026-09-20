"""Planner: produce a bounded, ordered plan of steps from a selected hypothesis.

Each :class:`PlanStep` names a tool to call and the phase that gates it. The
planner is deterministic: the same hypothesis yields the same plan. The risk of
the eventual fix determines whether an approval gate is inserted (the policy
engine makes that call, not the planner).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models import Hypothesis, RiskLevel
from app.models.state import Phase


@dataclass
class PlanStep:
    tool: str
    phase: Phase
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class Planner:
    def build_plan(self, hypothesis: Hypothesis) -> list[PlanStep]:
        hkind = hypothesis.id.split(":", 1)[0]

        steps: list[PlanStep] = [
            PlanStep("get_expected_schema", Phase.DETECT, {}, "read expected schema"),
            PlanStep("get_observed_schema", Phase.DETECT, {}, "infer observed schema"),
            PlanStep("compare_schemas", Phase.DETECT, {}, "detect drift events"),
            PlanStep("get_lineage", Phase.DIAGNOSE, {}, "assess blast radius"),
            PlanStep("profile_dataset", Phase.DIAGNOSE, {}, "profile observed data"),
            PlanStep("sample_rows", Phase.DIAGNOSE, {}, "sample representative values"),
        ]

        if hkind == "type_drift":
            steps.extend(
                [
                    PlanStep("propose_transformation", Phase.PROPOSE, {}, "propose cast fix"),
                    PlanStep("validate_transformation", Phase.VALIDATE, {}, "shadow-validate"),
                    PlanStep("create_pipeline_version", Phase.STAGE, {}, "persist new version"),
                    PlanStep("stage_pipeline", Phase.STAGE, {}, "stage candidate"),
                    PlanStep("run_canary", Phase.CANARY, {}, "run 95/5 canary"),
                    PlanStep("monitor_canary", Phase.MONITOR, {}, "check health"),
                ]
            )
        elif hkind == "nulls":
            steps.extend(
                [
                    PlanStep("propose_transformation", Phase.PROPOSE, {}, "propose fill fix"),
                    PlanStep("validate_transformation", Phase.VALIDATE, {}, "shadow-validate"),
                    PlanStep("create_pipeline_version", Phase.STAGE, {}, "persist new version"),
                    PlanStep("stage_pipeline", Phase.STAGE, {}, "stage candidate"),
                    PlanStep("run_canary", Phase.CANARY, {}, "run canary"),
                    PlanStep("monitor_canary", Phase.MONITOR, {}, "check health"),
                ]
            )
        else:
            # Investigation-only default: surface evidence, do not auto-mutate.
            steps.append(
                PlanStep("compare_distributions", Phase.DIAGNOSE, {}, "compare distributions")
            )

        return steps

    def needs_approval(self, plan: list[PlanStep], risk: RiskLevel) -> bool:
        # Staging/canary steps gated by policy; return whether approval is needed.
        if risk == RiskLevel.HIGH:
            return True
        if risk == RiskLevel.MEDIUM and any(
            s.tool in ("create_pipeline_version", "stage_pipeline", "run_canary")
            for s in plan
        ):
            return True
        return False
