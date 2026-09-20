"""Tool abstraction.

Each tool is a strongly-typed callable returning structured output (dicts or
Pydantic models). Tools are gated by the :class:`PolicyEngine` *before* dispatch;
an agent/LLM cannot invoke a tool that the policy doesn't permit for the current
phase.
"""

from __future__ import annotations

import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from app.agent.policies import ActionClass

ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    action: ActionClass
    handler: ToolHandler
    description: str = ""

    def __get__(self, instance: Any, owner: Any) -> "Tool":
        # Descriptor protocol: when accessed via an instance, bind `handler`
        # to that instance so the underlying method receives `self`.
        if instance is None:
            return self
        return replace(self, handler=types.MethodType(self.handler, instance))

    async def invoke(self, **kwargs: Any) -> Any:
        return await self.handler(**kwargs)


def tool(name: str, action: ActionClass, description: str = "") -> Callable[[ToolHandler], Tool]:
    def deco(fn: ToolHandler) -> Tool:
        return Tool(name=name, action=action, handler=fn, description=description)
    return deco
