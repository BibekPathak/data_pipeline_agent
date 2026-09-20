"""Pydantic domain models for the Self-Healing Data Pipeline Agent.

Split into focused modules so each concept stays small and testable:
- ``schema``  : column/schema definitions and drift events
- ``pipeline``: pipeline + stage definitions
- ``quality`` : quality metrics and quality anomalies
- ``fix``     : declarative transformations, fix proposals, validation reports
- ``state``   : agent triage state machine state and status enums
"""

from app.models.pipeline import Pipeline, PipelineStage
from app.models.quality import QualityAnomaly, QualityCheck, QualityReport
from app.models.schema import (
    ColumnDefinition,
    DataType,
    DriftEvent,
    DriftEventType,
    RenameHypothesis,
    SchemaDefinition,
    Severity,
)
from app.models.fix import (
    FixOperation,
    FixOperationType,
    FixProposal,
    OnError,
    RiskLevel,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from app.models.state import (
    ApprovalStatus,
    DeploymentStatus,
    Hypothesis,
    Phase,
    PipelineTriageState,
)

__all__ = [
    "Pipeline",
    "PipelineStage",
    "QualityAnomaly",
    "QualityCheck",
    "QualityReport",
    "ColumnDefinition",
    "DataType",
    "DriftEvent",
    "DriftEventType",
    "RenameHypothesis",
    "SchemaDefinition",
    "Severity",
    "FixOperation",
    "FixOperationType",
    "FixProposal",
    "OnError",
    "RiskLevel",
    "ValidationCheck",
    "ValidationReport",
    "ValidationStatus",
    "ApprovalStatus",
    "DeploymentStatus",
    "Hypothesis",
    "Phase",
    "PipelineTriageState",
]
