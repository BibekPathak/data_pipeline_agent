"""Tool registry: assembles all tools around a ToolContext and validates them.

Enforces that every registered tool name exists in the policy engine's action
mapping — a tool with no safety classification cannot be added to the registry
(belt-and-braces against an unclassified tool being passed to the agent).
"""

from __future__ import annotations

from app.agent.policies import ActionClass, PolicyEngine, TOOL_ACTION_CLASS
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.data_tools import DataTools
from app.tools.deployment_tools import DeploymentTools
from app.tools.fix_tools import FixTools
from app.tools.pipeline_tools import PipelineTools
from app.tools.schema_tools import SchemaTools


class ToolRegistry:
    def __init__(self, ctx: ToolContext, policy: PolicyEngine) -> None:
        self.ctx = ctx
        self.policy = policy
        self._tools: dict[str, Tool] = {}

        for tc in (
            SchemaTools(ctx),
            DataTools(ctx),
            PipelineTools(ctx),
            FixTools(ctx),
            DeploymentTools(ctx),
        ):
            for t in tc.tools():
                self._register(t)

    def _register(self, t: Tool) -> None:
        if t.name not in TOOL_ACTION_CLASS:
            raise ValueError(
                f"Tool {t.name!r} has no safety classification in the policy engine"
            )
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def invoke(self, name: str, phase, approval=None, **kwargs):
        """Safely invoke a tool: policy-gate then dispatch. Structured result."""
        self.policy.assert_action_allowed(name, phase, approval)
        t = self._tools.get(name)
        if t is None:
            raise KeyError(f"No tool named {name!r}")
        return t.handler(**kwargs)
