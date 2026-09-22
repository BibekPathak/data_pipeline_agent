"""Evaluation reports: markdown + JSON summaries of a scenario run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.metrics import EvalMetrics, compute_metrics
from app.evaluation.models import ScenarioResult


def build_report_data(
    results: list[ScenarioResult], metrics: EvalMetrics
) -> dict[str, Any]:
    return {
        "metrics": metrics.to_dict(),
        "scenarios": [r.to_dict() for r in results],
    }


def to_markdown(results: list[ScenarioResult], metrics: EvalMetrics) -> str:
    d = metrics.to_dict()
    lines: list[str] = [
        "# Self-Healing Pipeline Agent — Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    labels = {
        "total_scenarios": "Total scenarios",
        "outcome_match_rate": "Outcome match rate",
        "drift_detection_rate": "Drift detection rate",
        "root_cause_accuracy": "Root cause accuracy",
        "fix_success_rate": "Fix success rate",
        "validation_pass_rate": "Validation pass rate",
        "rollback_success_rate": "Rollback success rate",
        "data_loss_rate": "DATA LOSS RATE (target 0)",
        "false_repair_rate": "False repair rate",
        "unnecessary_change_rate": "Unnecessary change rate",
        "average_tool_calls": "Avg tool calls",
        "average_latency_s": "Avg latency (s)",
        "estimated_cost_usd": "Estimated cost (USD)",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {d.get(key)} |")
    lines += ["", "## Scenarios", "", "| Scenario | Expected | Matched | Diagnosis | Fix | Data loss |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.scenario.id} | {r.scenario.expected_outcome.value} "
            f"| {'YES' if r.matched_outcome else 'NO'} "
            f"| {'YES' if r.correct_diagnosis else 'no'} "
            f"| {'YES' if r.safe_fix_proposed else 'no'} "
            f"| {r.data_loss} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(
    results: list[ScenarioResult],
    out_dir: str | Path = "./reports",
) -> tuple[Path, Path]:
    metrics = compute_metrics(results)
    data = build_report_data(results, metrics)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "evaluation.md"
    json_path = out / "evaluation.json"
    md_path.write_text(to_markdown(results, metrics))
    json_path.write_text(json.dumps(data, indent=2, default=str))
    return md_path, json_path
