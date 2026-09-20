"""Rollback package: deterministic version restoration."""

from app.rollback.manager import RollbackManager, RollbackReport

__all__ = ["RollbackManager", "RollbackReport"]
