from __future__ import annotations

from geoharness.registry import ToolRegistry, call_handler
from geoharness.schemas import Diagnostic, GeoSkillCall, GeoSkillResult
from geoharness.store import ArtifactStore


class GeoHarnessRuntime:
    def __init__(self, store: ArtifactStore, registry: ToolRegistry) -> None:
        self.store = store
        self.registry = registry

    def execute(self, call: GeoSkillCall) -> GeoSkillResult:
        registered_tool = self.registry.get(call.tool_name)
        if registered_tool is None:
            return GeoSkillResult(
                status="failed",
                diagnostics=[
                    Diagnostic(
                        code="unknown_tool",
                        severity="fatal",
                        message=f"Unknown tool: {call.tool_name}",
                    )
                ],
            )

        try:
            return call_handler(registered_tool.handler, self.store, call.arguments)
        except Exception as exc:
            return GeoSkillResult(
                status="failed",
                diagnostics=[
                    Diagnostic(
                        code="tool_execution_error",
                        severity="fatal",
                        message=str(exc),
                    )
                ],
            )
