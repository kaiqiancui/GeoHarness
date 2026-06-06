from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


ArtifactType = Literal["raster", "vector", "table", "json", "report"]
Severity = Literal["info", "warning", "fatal"]
ResultStatus = Literal["success", "warning", "failed"]


@dataclass
class GeoSkillSpec:
    name: str
    family: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    preconditions: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeoSkillCall:
    tool_name: str
    arguments: dict[str, Any]
    expected_output_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnostic:
    code: str
    severity: Severity
    message: str
    artifact_id: str | None = None
    check_name: str | None = None
    measured_value: Any | None = None
    threshold: Any | None = None
    suggested_actions: list[str] = field(default_factory=list)
    agent_visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeoArtifact:
    id: str
    type: ArtifactType
    path: str
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    resolution: tuple[float, float] | None = None
    transform: tuple[float, ...] | None = None
    shape: tuple[int, ...] | None = None
    bands: list[str] | None = None
    nodata: float | int | None = None
    timestamp: str | None = None
    sensor: str | None = None
    parents: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(Path(self.path))
        return data


@dataclass
class GeoSkillResult:
    status: ResultStatus
    artifacts: list[GeoArtifact] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_recovery: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "warnings": self.warnings,
            "suggested_recovery": self.suggested_recovery,
            "provenance": self.provenance,
        }


def status_from_diagnostics(diagnostics: list[Diagnostic]) -> ResultStatus:
    if any(item.severity == "fatal" for item in diagnostics):
        return "failed"
    if any(item.severity == "warning" for item in diagnostics):
        return "warning"
    return "success"
