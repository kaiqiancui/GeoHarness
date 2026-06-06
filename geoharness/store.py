from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geoharness.schemas import Diagnostic, GeoArtifact


DiagnosticKey = tuple[str, str, str | None, str | None, str, str]


def diagnostic_key(diagnostic: Diagnostic) -> DiagnosticKey:
    """Stable identity for repeated checks of the same artifact."""

    return (
        diagnostic.code,
        diagnostic.severity,
        diagnostic.artifact_id,
        diagnostic.check_name,
        _stable_json(diagnostic.measured_value),
        _stable_json(diagnostic.threshold),
    )


def deduplicate_diagnostics(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    seen: set[DiagnosticKey] = set()
    unique: list[Diagnostic] = []
    for diagnostic in diagnostics:
        key = diagnostic_key(diagnostic)
        if key in seen:
            continue
        seen.add(key)
        unique.append(diagnostic)
    return unique


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class ArtifactStore:
    """Local filesystem artifact store with JSON metadata."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.metadata_path = self.root / "metadata.json"
        self.diagnostics_path = self.root / "diagnostics.jsonl"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, GeoArtifact] = {}
        self._recorded_diagnostic_keys: set[DiagnosticKey] = set()
        if self.metadata_path.exists():
            raw = json.loads(self.metadata_path.read_text())
            self._artifacts = {
                artifact_id: GeoArtifact(**payload)
                for artifact_id, payload in raw.get("artifacts", {}).items()
            }
        if self.diagnostics_path.exists():
            for line in self.diagnostics_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                self._recorded_diagnostic_keys.add(diagnostic_key(Diagnostic(**json.loads(line))))

    def artifact_path(self, artifact_id: str, suffix: str) -> Path:
        return self.artifacts_dir / f"{artifact_id}{suffix}"

    def add(self, artifact: GeoArtifact) -> GeoArtifact:
        self._artifacts[artifact.id] = artifact
        self.flush()
        return artifact

    def get(self, artifact_id: str) -> GeoArtifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact id: {artifact_id}") from exc

    def all(self) -> list[GeoArtifact]:
        return list(self._artifacts.values())

    def record_diagnostics(self, diagnostics: list[Diagnostic]) -> None:
        if not diagnostics:
            return
        unique_diagnostics = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic_key(diagnostic) not in self._recorded_diagnostic_keys
        ]
        if not unique_diagnostics:
            return
        with self.diagnostics_path.open("a", encoding="utf-8") as handle:
            for diagnostic in unique_diagnostics:
                self._recorded_diagnostic_keys.add(diagnostic_key(diagnostic))
                handle.write(json.dumps(diagnostic.to_dict(), ensure_ascii=False) + "\n")

    def flush(self) -> None:
        payload = {
            "artifacts": {
                artifact_id: artifact.to_dict()
                for artifact_id, artifact in sorted(self._artifacts.items())
            }
        }
        self.metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
