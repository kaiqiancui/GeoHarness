from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from geoharness.schemas import GeoSkillResult, GeoSkillSpec
from geoharness.store import ArtifactStore


GeoSkillHandler = Callable[..., GeoSkillResult]


@dataclass(frozen=True)
class RegisteredTool:
    spec: GeoSkillSpec
    handler: GeoSkillHandler


class ToolRegistry:
    """Registry for deterministic GeoSkill discovery and execution lookup."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: GeoSkillSpec, handler: GeoSkillHandler) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[GeoSkillSpec]:
        return [self._tools[name].spec for name in sorted(self._tools)]


def call_handler(handler: GeoSkillHandler, store: ArtifactStore, arguments: dict) -> GeoSkillResult:
    """Invoke a handler using the GeoHarness convention: handler(store, **arguments)."""

    return handler(store, **arguments)
