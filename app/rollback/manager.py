"""Rollback manager: deterministic, idempotent version restoration.

Rollback treats the warehouse as versioned (table, version) snapshots. Before
promotion the manager records ``active`` and ``candidate`` versions; on failure
or health decline it restores the previous active version and audits the event.

Deliberately idempotent: restoring the same version twice yields the same result,
and a rollback after an already-restored state is a no-op.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from app.storage.base import Warehouse


@dataclass
class RollbackReport:
    run_id: str
    rolled_back: bool
    reason: str | None = None
    previous_version: str | None = None
    restored_version: str | None = None
    failed_metrics: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RollbackManager:
    def __init__(self, warehouse: Warehouse, metadata=None) -> None:
        self._warehouse = warehouse
        self._metadata = metadata
        self._active: dict[str, str] = {}  # table -> active version
        self._previous: dict[str, str] = {}  # table -> previous version

    def set_active(self, table: str, version: str) -> None:
        self._previous[table] = self._active.get(table)
        self._active[table] = version

    def active_version(self, table: str) -> str | None:
        return self._active.get(table)

    async def rollback(
        self,
        run_id: str,
        table: str,
        *,
        reason: str = "",
        failed_metrics: dict | None = None,
    ) -> RollbackReport:
        prev = self._previous.get(table)
        active = self._active.get(table)
        if prev is None or active is None or prev == active:
            # Nothing to restore, or already rolled back -> idempotent no-op.
            return RollbackReport(
                run_id=run_id,
                rolled_back=False,
                reason="no previous version to restore (idempotent no-op)",
                previous_version=active,
                restored_version=active,
                failed_metrics=failed_metrics or {},
            )

        # Restore: read the previous known-good version and write it as active.
        df = await self._warehouse.read_table(table, f"main:{prev}")
        if df is not None:
            await self._warehouse.write_table(table, f"main:{prev}", df)
        self._active[table] = prev
        self._previous[table] = None

        report = RollbackReport(
            run_id=run_id,
            rolled_back=True,
            reason=reason,
            previous_version=active,
            restored_version=prev,
            failed_metrics=failed_metrics or {},
        )
        if self._metadata is not None:
            await self._metadata.record_rollback(
                {
                    "run_id": run_id,
                    "reason": reason,
                    "previous_version": active,
                    "restored_version": prev,
                    "timestamp": report.timestamp,
                    "failed_metrics": failed_metrics or {},
                }
            )
        return report
