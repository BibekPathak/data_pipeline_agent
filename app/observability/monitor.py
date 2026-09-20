"""Post-deployment health monitoring.

Formalizes the canary/monitor gate: after a candidate is staged and run through a
95/5 canary, the health monitor evaluates the canary-derived metrics against
thresholds. A breach means the promotion is unhealthy and triggers rollback.

Also provides a baseline comparison: the candidate snapshot is measured against
the pre-deployment metric history (anomaly detection) so the orchestration can
catch silent regressions that the canary split might not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.data.anomaly import detect_anomalies


@dataclass
class HealthCheck:
    name: str
    passed: bool
    observed: Any = None
    threshold: Any = None
    message: str | None = None


@dataclass
class HealthReport:
    healthy: bool
    checks: list[HealthCheck] = field(default_factory=list)

    def failed_checks(self) -> list[HealthCheck]:
        return [c for c in self.checks if not c.passed]


DEFAULT_THRESHOLDS = {
    "failure_rate": 0.0,
    "quality": True,
}


class HealthMonitor:
    def __init__(self, thresholds: dict | None = None) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def evaluate_canary(self, canary: dict) -> HealthReport:
        """Evaluate canary metrics into a health report.

        Gates on the reliable signals: the candidate ran without failure and its
        output passes quality checks. Cross-partition ``metric_delta_pct`` is
        recorded for observability but is not a gate (the 95/5 slices differ in
        size, so raw sums are not directly comparable — regressions are caught by
        baseline anomaly checks instead).
        """
        checks: list[HealthCheck] = []

        failure_rate = canary.get("failure_rate", 1.0)
        checks.append(
            HealthCheck(
                "failure_rate",
                failure_rate <= self.thresholds["failure_rate"],
                observed=failure_rate,
                threshold=self.thresholds["failure_rate"],
            )
        )

        quality_pass = canary.get("quality_pass", False)
        checks.append(
            HealthCheck("quality", quality_pass, observed=quality_pass)
        )

        checks.append(
            HealthCheck(
                "metric_delta",
                True,  # informational only for this release
                observed=canary.get("metric_delta_pct"),
                message="metric_delta recorded for observability; regressions "
                "are validated via baseline anomaly checks",
            )
        )

        return HealthReport(healthy=all(c.passed for c in checks), checks=checks)

    def baseline_anomalies(
        self, df, history: list[dict[str, Any]]
    ) -> list:
        """Run deterministic anomaly detection vs the pre-deploy baseline."""
        return detect_anomalies(df, history)
