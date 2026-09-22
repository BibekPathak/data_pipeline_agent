"""Storage package: protocols, SQLite + in-memory backends, and a factory."""

from __future__ import annotations

from pathlib import Path

from app.storage.base import BackendStore, MetadataStore, Warehouse
from app.storage.memory import MemoryMetadataStore, MemoryWarehouse
from app.storage.sqlite import SQLiteMetadataStore, SQLiteWarehouse


def create_store(
    backend: str = "sqlite",
    db_path: str | None = None,
    *,
    create: bool = True,
) -> BackendStore:
    """Build a combined metadata + warehouse store.

    ``backend`` maps to the concrete implementations:
      - "sqlite": SQLite file (default)
      - "memory": in-memory (tests / fast deterministic runs)
    """
    if backend == "memory":
        return BackendStore(MemoryMetadataStore(), MemoryWarehouse())

    if backend == "sqlite":
        path = db_path or "./data/pipeline.db"
        # SQLite does not create missing parent directories; do it defensively
        # so a fresh clone (where data/ is not tracked by git) works.
        parent = Path(path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        metadata = SQLiteMetadataStore(path, create=create)
        warehouse = SQLiteWarehouse(path, create=create)
        return BackendStore(metadata, warehouse)

    raise ValueError(f"Unknown storage backend: {backend!r}")


def store_from_settings(settings: object = None) -> BackendStore:
    """Create a store from a :class:`app.config.Settings`-compatible object.

    Pass nothing to use the module-level default settings; pass an override to
    inject a custom backend/path (used in tests).
    """
    if settings is None:
        from app.config import settings as _settings

        settings = _settings
    return create_store(
        backend=settings.storage_backend.value,
        db_path=str(settings.db_path),
    )


__all__ = [
    "BackendStore",
    "MetadataStore",
    "Warehouse",
    "MemoryMetadataStore",
    "MemoryWarehouse",
    "SQLiteMetadataStore",
    "SQLiteWarehouse",
    "create_store",
    "store_from_settings",
]
