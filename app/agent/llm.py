"""LLM abstraction for the agent.

Defines :class:`LLMProviderProtocol` and two implementations:

- :class:`DeterministicLLM` — offline, rule-based reproduction of structured
  proposals. Used by default and by evaluation so every scenario is fully
  reproducible without an API key. It translates deterministic evidence
  (drift events + quality anomalies) into a structured :class:`FixProposal`.
- :class:`OpenAIProvider` — thin wrapper over the OpenAI SDK implementing the
  same protocol. Tested via a mock; live use requires an API key in `.env`.

The agent never executes free-form LLM output. The LLM only *produces* a
declarative :class:`FixProposal`; the policy + staging layers decide whether
and how it runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from app.models import (
    DataType,
    DriftEvent,
    DriftEventType,
    FixOperation,
    FixOperationType,
    FixProposal,
    OnError,
    QualityAnomaly,
    RiskLevel,
    Severity,
)


@runtime_checkable
class LLMProviderProtocol(Protocol):
    async def build_proposal(
        self,
        *,
        drift_events: list[DriftEvent],
        quality_anomalies: list[QualityAnomaly],
        hypotheses: list[str],
        pipeline_context: dict[str, Any],
    ) -> FixProposal | None: ...

    def name(self) -> str: ...


class LLMProvider(ABC):
    @abstractmethod
    async def build_proposal(self, **kwargs: Any) -> FixProposal | None:
        raise NotImplementedError


# --- Deterministic, rule-based implementation ---


def _propose_cast(event: DriftEvent) -> FixOperation:
    # Cast the drifted (observed) type BACK to the expected type.
    target = event.expected or event.observed
    if target in (None, DataType.UNKNOWN.value):
        target = "DOUBLE"
    return FixOperation(
        operation=FixOperationType.CAST_TYPE,
        column=event.column,
        to_type=target,
        on_error=OnError.NULL,
    )


class DeterministicLLM(LLMProvider):
    """Rule-based proposal builder; no network, fully deterministic.

    Maps each detected issue to the minimal safe declarative operation, and
    infers risk/confidence from severity and data-loss potential. Used to keep
    scenarios reproducible and to guarantee the system runs without a key.
    """

    def name(self) -> str:
        return "deterministic"

    async def build_proposal(
        self,
        *,
        drift_events: list[DriftEvent],
        quality_anomalies: list[QualityAnomaly],
        hypotheses: list[str],
        pipeline_context: dict[str, Any],
    ) -> FixProposal | None:
        ops: list[FixOperation] = []
        evidence: list[str] = []

        for e in drift_events:
            evidence.append(
                f"{e.column}: {e.expected} -> {e.observed} ({e.type.value})"
            )
            if e.type == DriftEventType.TYPE_CHANGED:
                ops.append(_propose_cast(e))
            elif e.type == DriftEventType.COLUMN_ADDED:
                # Non-breaking; no operation required for the pipeline to run.
                evidence.append(f"new column {e.column} tolerated: NULLABLE")
            elif e.type == DriftEventType.COLUMN_REMOVED:
                # Blocking drift: cannot silently fabricate a column.
                evidence.append(
                    f"column {e.column} removed; propose drop_invalid_rows only "
                    "after explicit confirmation"
                )
            elif e.type == DriftEventType.NULLABLE_CHANGED:
                evidence.append(f"nullability change on {e.column} acceptable")

        for a in quality_anomalies:
            evidence.append(
                f"{a.metric}:{a.column or ''} observed={a.observed} expected={a.expected}"
            )
            if a.metric == "duplicate_rate" and a.observed and a.observed > 0.01:
                ops.append(FixOperation(operation=FixOperationType.DEDUPLICATE))
            elif a.metric == "null_rate" and a.column:
                ops.append(
                    FixOperation(
                        operation=FixOperationType.DEFAULT_VALUE,
                        column=a.column,
                        value=0,
                    )
                )
            elif a.metric == "referential_integrity" and a.column:
                ops.append(
                    FixOperation(
                        operation=FixOperationType.DROP_INVALID_ROWS,
                        column=a.column,
                    )
                )

        if not ops:
            return None

        # Risk: block (deletion-heavy) fixes from auto-run unless explicit.
        has_drop = any(
            o.operation
            in (FixOperationType.DROP_INVALID_ROWS, FixOperationType.DEDUPLICATE)
            for o in ops
        )
        has_cast = any(o.operation == FixOperationType.CAST_TYPE for o in ops)
        if has_drop and has_cast:
            risk = RiskLevel.HIGH
            confidence = 0.6
        elif has_drop:
            risk = RiskLevel.MEDIUM
            confidence = 0.7
        else:
            risk = RiskLevel.LOW
            confidence = 0.92

        root_cause = (
            "Schema drift detected ("
            + "; ".join(evidence[:4])
            + ")"
            if drift_events
            else "Data-quality anomalies require remediation"
        )

        return FixProposal(
            root_cause=root_cause,
            evidence=evidence,
            operations=ops,
            affected_stages=list(pipeline_context.get("affected_stages", [])),
            affected_columns=sorted({o.column for o in ops if o.column}),
            expected_impact="Brings observed data into expected schema and quality.",
            risk=risk,
            confidence=confidence,
            validation_plan=[
                "schema",
                "row_count",
                "null_rate",
                "duplicate_rate",
                "aggregates",
            ],
            rollback_strategy="restore previous active pipeline version",
        )


# --- OpenAI implementation (tested via mock) ---


class OpenAIProvider(LLMProvider):
    """OpenAI-backed proposal builder implementing the same protocol.

    Structured outputs are requested via ``response_format=json_object`` and
    validated/mapped into a :class:`FixProposal`. Only *known* operation types
    are accepted; anything else is discarded (never executed).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    def name(self) -> str:
        return "openai"

    async def build_proposal(self, **kwargs: Any) -> FixProposal | None:
        drift_events: list[DriftEvent] = kwargs.get("drift_events", [])
        quality_anomalies: list[QualityAnomaly] = kwargs.get("quality_anomalies", [])
        hypotheses: list[str] = kwargs.get("hypotheses", [])
        pipeline_context: dict[str, Any] = kwargs.get("pipeline_context", {})

        system = (
            "You are a cautious data infrastructure agent. Given schema drift and "
            "quality anomalies, produce a JSON FixProposal using ONLY these "
            "operation types: cast_type, rename_column, fill_null, drop_invalid_rows, "
            "normalize_string, parse_timestamp, deduplicate, default_value, "
            "column_mapping. Never propose arbitrary SQL or table drops. Risk is "
            "LOW/MEDIUM/HIGH."
        )
        user = {
            "drift_events": [e.model_dump() for e in drift_events],
            "quality_anomalies": [a.model_dump() for a in quality_anomalies],
            "hypotheses": hypotheses,
            "pipeline_context": {"affected_stages": pipeline_context.get("affected_stages", [])},
        }
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(user)},
                ],
            )
            raw = resp.choices[0].message.content
            return _parse_proposal_json(raw)
        except Exception:
            return None


def _parse_proposal_json(raw: str | None) -> FixProposal | None:
    """Parse + sanitize an OpenAI JSON response into a FixProposal."""
    import json

    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    ops: list[FixOperation] = []
    for o in data.get("operations", []):
        try:
            op = FixOperation(
                operation=FixOperationType(o.get("operation")),
                column=o.get("column"),
                target=o.get("target"),
                to_type=o.get("to_type"),
                value=o.get("value"),
                on_error=OnError(o.get("on_error", "reject")),
            )
            ops.append(op)
        except (ValueError, TypeError):
            continue

    try:
        return FixProposal(
            root_cause=data.get("root_cause", ""),
            evidence=data.get("evidence", []),
            operations=ops,
            affected_stages=data.get("affected_stages", []),
            affected_columns=data.get("affected_columns", []),
            expected_impact=data.get("expected_impact"),
            risk=RiskLevel(data.get("risk", "LOW")),
            confidence=float(data.get("confidence", 0.0)),
            validation_plan=data.get("validation_plan", []),
            rollback_strategy=data.get("rollback_strategy"),
        )
    except ValueError:
        return None


def create_llm(provider: object, settings: object | None = None) -> LLMProvider | None:
    """Create the LLM provider selected by config.

    ``provider`` is an ``LLMProviderKind``-compatible enum or a string. Returns
    None for ``openai`` when no API key is configured.
    """
    from app.config import LLMProviderKind

    try:
        kind = LLMProviderKind(provider.value if hasattr(provider, "value") else provider)
    except ValueError:
        kind = LLMProviderKind.DETERMINISTIC

    if kind == LLMProviderKind.OPENAI:
        if settings is None:
            from app.config import settings as _s

            settings = _s
        key = getattr(settings, "openai_api_key", "") or ""
        if not key:
            return None
        return OpenAIProvider(api_key=key, model=getattr(settings, "openai_model", "gpt-4o-mini"))

    return DeterministicLLM()
