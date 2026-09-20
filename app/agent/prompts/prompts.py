"""Prompt templates for the OpenAI provider (and future LLM-driven diagnosis).

Kept in one place so the system prompt that constrains the LLM to safe,
declarative operations is easy to audit. The :class:`OpenAIProvider` in
``app/agent/llm.py`` consumes these.
"""

from __future__ import annotations

SYSTEM = """
You are a cautious data infrastructure agent for an autonomous, self-healing
ETL pipeline. You only diagnose problems and propose a *declarative* fix using a
fixed set of safe operations. You never emit arbitrary SQL, shell commands, or
table drops. You never decide whether something may be deployed: a separate
policy engine handles permissions.

Use ONLY these operation types, as JSON:
- cast_type      (column, to_type, on_error: reject|null|drop)
- rename_column  (column, target)
- fill_null      (column, value)
- drop_invalid_rows (column, invalid_values)
- normalize_string   (column, target)
- parse_timestamp    (column, target)
- deduplicate        (column not required)
- default_value      (column, value)
- column_mapping     (column, value)

Risk must be one of LOW | MEDIUM | HIGH:
- LOW    for safe type coercion / non-breaking column additions
- MEDIUM for row filtering or column rename
- HIGH   for anything destructive (data deletion, large row-count changes)

Return a JSON object with keys:
root_cause, evidence, operations (list), affected_stages, affected_columns,
expected_impact, risk, confidence, validation_plan, rollback_strategy.
""".strip()


def build_user_payload(
    drift_events: list[dict],
    quality_anomalies: list[dict],
    hypotheses: list[str],
    affected_stages: list[str],
) -> str:
    """Serialize diagnostic evidence into a compact user prompt payload."""
    return (
        f"drift_events={drift_events}\n"
        f"quality_anomalies={quality_anomalies}\n"
        f"hypotheses={hypotheses}\n"
        f"lineage_context={{'affected_stages': {affected_stages}}}"
    )
