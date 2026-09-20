"""Tests for the LLM providers: DeterministicLLM and OpenAIProvider (mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.llm import DeterministicLLM, OpenAIProvider, create_llm
from app.models import (
    DataType,
    DriftEvent,
    DriftEventType,
    FixOperationType,
    QualityAnomaly,
    RiskLevel,
)
from app.agent import policies  # noqa: F401 (ensure policy package imports)


def _type_drift() -> DriftEvent:
    return DriftEvent(
        type=DriftEventType.TYPE_CHANGED,
        column="amount",
        expected=DataType.DOUBLE.value,
        observed=DataType.VARCHAR.value,
    )


class TestDeterministicLLM:
    @pytest.mark.asyncio
    async def test_type_drift_proposes_cast(self):
        llm = DeterministicLLM()
        proposal = await llm.build_proposal(
            drift_events=[_type_drift()],
            quality_anomalies=[],
            hypotheses=[],
            pipeline_context={},
        )
        assert proposal is not None
        assert proposal.operations[0].operation == FixOperationType.CAST_TYPE
        assert proposal.operations[0].column == "amount"
        assert proposal.operations[0].to_type == "DOUBLE"
        assert proposal.risk == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_null_spike_proposes_default_value(self):
        llm = DeterministicLLM()
        proposal = await llm.build_proposal(
            drift_events=[],
            quality_anomalies=[
                QualityAnomaly(metric="null_rate", column="amount", observed=0.22)
            ],
            hypotheses=[],
            pipeline_context={},
        )
        assert proposal is not None
        assert any(
            o.operation == FixOperationType.DEFAULT_VALUE and o.column == "amount"
            for o in proposal.operations
        )

    @pytest.mark.asyncio
    async def test_no_evidence_returns_none(self):
        llm = DeterministicLLM()
        assert llm.name() == "deterministic"
        proposal = await llm.build_proposal(
            drift_events=[], quality_anomalies=[], hypotheses=[], pipeline_context={}
        )
        assert proposal is None


class TestOpenAIProviderMocked:
    @pytest.mark.asyncio
    async def test_parses_structured_proposal(self, monkeypatch):
        # Bypass __init__ to avoid real client construction, then inject a mock.
        prov = object.__new__(OpenAIProvider)
        prov._model = "gpt-test"
        client = MagicMock()
        message = MagicMock()
        message.content = (
            '{"root_cause": "type drift", "operations": '
            '[{"operation": "cast_type", "column": "amount", "to_type": "DOUBLE", '
            '"on_error": "null"}], "risk": "LOW", "confidence": 0.9}'
        )
        resp = MagicMock()
        resp.choices = [MagicMock(message=message)]
        client.chat.completions.create = AsyncMock(return_value=resp)
        prov._client = client

        proposal = await prov.build_proposal(
            drift_events=[_type_drift()],
            quality_anomalies=[],
            hypotheses=[],
            pipeline_context={},
        )
        assert proposal is not None
        assert proposal.operations[0].to_type == "DOUBLE"
        assert proposal.risk == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_drops_unknown_operation(self, monkeypatch):
        prov = object.__new__(OpenAIProvider)
        prov._model = "gpt-test"
        client = MagicMock()
        message = MagicMock()
        message.content = (
            '{"root_cause": "x", "operations": '
            '[{"operation": "DROP TABLE orders"}, {"operation": "cast_type", '
            '"column": "a", "to_type": "DOUBLE"}]}'
        )
        resp = MagicMock()
        resp.choices = [MagicMock(message=message)]
        client.chat.completions.create = AsyncMock(return_value=resp)
        prov._client = client

        proposal = await prov.build_proposal(
            drift_events=[], quality_anomalies=[], hypotheses=[], pipeline_context={}
        )
        assert proposal is not None
        # Unknown "DROP TABLE" operation is dropped, cast kept.
        assert all(o.operation == FixOperationType.CAST_TYPE for o in proposal.operations)


class TestLLMFactory:
    def test_creates_deterministic(self):
        from app.config import LLMProviderKind

        llm = create_llm(LLMProviderKind.DETERMINISTIC)
        from app.agent.llm import DeterministicLLM

        assert isinstance(llm, DeterministicLLM)

    def test_openai_without_key_returns_none(self):
        from app.config import LLMProviderKind

        settings = MagicMock()
        settings.openai_api_key = ""
        settings.openai_model = "gpt"
        assert create_llm(LLMProviderKind.OPENAI, settings) is None
