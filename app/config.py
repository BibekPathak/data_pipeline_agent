"""Application configuration via pydantic-settings.

Every tunable is read from environment variables (prefix ``SHP_``) or an
optional ``.env`` file. This keeps provider choice, agent guardrails, approval
mode and storage backend declarative and environment-driven.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderKind(str, Enum):
    DETERMINISTIC = "deterministic"
    OPENAI = "openai"


class ApprovalMode(str, Enum):
    AUTO = "auto"              # LOW auto-stages; unsupported/HIGH rejected without gate
    MANUAL = "manual"          # MEDIUM/HIGH require explicit approval (harness supplies it)
    HUMAN_IN_LOOP = "human_in_loop"  # MEDIUM/HIGH pause for a real human


class StorageBackend(str, Enum):
    SQLITE = "sqlite"
    MEMORY = "memory"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SHP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: LLMProviderKind = LLMProviderKind.DETERMINISTIC
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Agent budgets / guardrails ---
    max_iterations: int = 12
    max_tool_calls: int = 40
    execution_timeout_seconds: float = 300.0
    max_budget_rows: int = 2_000_000
    max_budget_bytes: int = 256 * 1024 * 1024

    # --- Approval ---
    approval_mode: ApprovalMode = ApprovalMode.AUTO

    # --- Storage ---
    storage_backend: StorageBackend = StorageBackend.SQLITE
    db_path: Path = Path("./data/pipeline.db")

    # --- Data / pipeline dirs ---
    pipeline_dir: Path = Path("./pipelines")
    fixture_dir: Path = Path("./datasets/fixtures")

    @property
    def data_dir(self) -> Path:
        return self.db_path.parent


settings = Settings()
