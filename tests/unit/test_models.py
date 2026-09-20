"""Unit tests for Phase 1: config settings and Pydantic domain models."""

from __future__ import annotations

from app.config import ApprovalMode, LLMProviderKind, Settings, StorageBackend
from app.models import (
    DataType,
    DriftEvent,
    DriftEventType,
    FixOperation,
    FixOperationType,
    FixProposal,
    OnError,
    Phase,
    Pipeline,
    PipelineStage,
    PipelineTriageState,
    RiskLevel,
    SchemaDefinition,
    Severity,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
    ColumnDefinition,
)


def _orders_schema(version: int = 3) -> SchemaDefinition:
    return SchemaDefinition(
        table="orders",
        version=version,
        columns=[
            ColumnDefinition(name="order_id", type=DataType.INTEGER),
            ColumnDefinition(name="amount", type=DataType.DOUBLE),
            ColumnDefinition(name="currency", type=DataType.VARCHAR),
        ],
    )


class TestSettings:
    def test_defaults(self):
        s = Settings(_env_file=None)
        assert s.llm_provider == LLMProviderKind.DETERMINISTIC
        assert s.approval_mode == ApprovalMode.AUTO
        assert s.storage_backend == StorageBackend.SQLITE
        assert s.max_iterations > 0
        assert s.max_tool_calls > 0

    def test_env_prefix_override(self, monkeypatch):
        monkeypatch.setenv("SHP_LLM_PROVIDER", "openai")
        monkeypatch.setenv("SHP_APPROVAL_MODE", "manual")
        s = Settings(_env_file=None)
        assert s.llm_provider == LLMProviderKind.OPENAI
        assert s.approval_mode == ApprovalMode.MANUAL


class TestSchemaDefinition:
    def test_column_map_and_names(self):
        sd = _orders_schema()
        assert set(sd.column_names()) == {"order_id", "amount", "currency"}
        assert sd.column_map["amount"].type == DataType.DOUBLE

    def test_add_version(self):
        sd = _orders_schema(version=4)
        assert sd.add_version().version == 5
        assert len(sd.add_version().columns) == 3


class TestPipeline:
    def test_stage_map(self):
        stage = PipelineStage(
            id="orders_clean",
            name="Orders Clean",
            input_schema=_orders_schema(),
            output_schema=_orders_schema(version=4),
            transformation="clean_orders",
        )
        p = Pipeline(
            id="orders",
            name="Orders Pipeline",
            version="1.0",
            source="fixtures:orders.csv",
            stages=[stage],
        )
        assert p.stage_map()["orders_clean"].name == "Orders Clean"
        assert p.base_input_schema().table == "orders"


class TestDriftEvent:
    def test_type_change(self):
        e = DriftEvent(
            type=DriftEventType.TYPE_CHANGED,
            column="amount",
            expected="DOUBLE",
            observed="VARCHAR",
            severity=Severity.HIGH,
        )
        assert e.type.value == "type_changed"
        assert e.severity == Severity.HIGH


class TestFixProposal:
    def test_cast_operation_roundtrip(self):
        op = FixOperation(
            operation=FixOperationType.CAST_TYPE,
            column="amount",
            from_type="VARCHAR",
            to_type="DOUBLE",
            on_error=OnError.REJECT,
        )
        assert op.operation.value == "cast_type"
        assert op.to_type == "DOUBLE"

    def test_mapping_fix(self):
        prop = FixProposal(
            root_cause="type drift",
            operations=[
                FixOperation(
                    operation=FixOperationType.CAST_TYPE,
                    column="amount",
                    to_type="DOUBLE",
                )
            ],
            affected_stages=["orders_clean"],
            risk=RiskLevel.LOW,
            confidence=0.95,
        )
        assert prop.risk == RiskLevel.LOW
        assert prop.confidence > 0.9


class TestValidationReport:
    def test_passed_all_green(self):
        r = ValidationReport(
            run_id="r1",
            checks=[
                ValidationCheck(name="row_count", status=ValidationStatus.PASSED),
                ValidationCheck(name="schema", status=ValidationStatus.PASSED),
            ],
        )
        assert r.passed is True
        assert r.failed == []

    def test_failed_detected(self):
        r = ValidationReport(
            run_id="r2",
            checks=[ValidationCheck(name="nulls", status=ValidationStatus.FAILED)],
        )
        assert r.passed is False
        assert len(r.failed) == 1


class TestAgentState:
    def test_defaults(self):
        s = PipelineTriageState(run_id="run1", pipeline_id="orders")
        assert s.phase == Phase.OBSERVE

    def test_bump_phase(self):
        s = PipelineTriageState(run_id="r", pipeline_id="orders")
        s.bump_phase(Phase.DETECT)
        assert s.phase == Phase.DETECT
        s.bump_phase(Phase.ROLLBACK)
        assert s.phase == Phase.ROLLBACK
