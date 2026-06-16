"""Context management — artifact summaries and LLM-facing tool result formatting.

The LLM cannot see raw rasters, so every tool result is distilled to a concise
JSON-serialisable summary that fits in the prompt window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ArtifactSummary:
    id: str
    type: str
    crs: str | None
    bounds: tuple[float, float, float, float] | None
    resolution: tuple[float, float] | None
    shape: tuple[int, ...] | None
    bands: list[str] | None
    quality: dict[str, Any]
    parents: list[str]
    provenance: dict[str, Any]


@dataclass
class ToolCallRecord:
    step: int
    tool_name: str
    arguments: dict[str, Any]
    status: str
    artifacts: list[ArtifactSummary]
    diagnostic_codes: list[str]
    warnings: list[str]


def artifact_summary(artifact: dict[str, Any]) -> ArtifactSummary:
    """Convert a GeoArtifact dict to a concise summary."""
    return ArtifactSummary(
        id=artifact.get("id", "?"),
        type=artifact.get("type", "?"),
        crs=artifact.get("crs"),
        bounds=_parse_bounds(artifact.get("bounds")),
        resolution=_parse_resolution(artifact.get("resolution")),
        shape=_parse_shape(artifact.get("shape")),
        bands=artifact.get("bands"),
        quality=artifact.get("quality") or {},
        parents=artifact.get("parents") or [],
        provenance=artifact.get("provenance") or {},
    )


def format_step_result(
    step: int,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> ToolCallRecord:
    """Build a structured record from a GeoSkillResult dict."""
    return ToolCallRecord(
        step=step,
        tool_name=tool_name,
        arguments=arguments,
        status=result.get("status", "unknown"),
        artifacts=[artifact_summary(a) for a in result.get("artifacts", [])],
        diagnostic_codes=[d.get("code", "") for d in result.get("diagnostics", [])],
        warnings=result.get("warnings", []),
    )


def format_tool_result_for_llm(
    tool_name: str,
    result: dict[str, Any],
) -> str:
    """Format a GeoSkillResult dict as a concise JSON string for the LLM."""
    status = result.get("status", "unknown")
    diagnostics = result.get("diagnostics", [])
    warnings = result.get("warnings", [])
    artifacts = result.get("artifacts", [])

    summary: dict[str, Any] = {
        "tool": tool_name,
        "status": status,
        "artifacts": [
            {
                "id": a.get("id"),
                "type": a.get("type"),
                "crs": a.get("crs"),
                "bounds": a.get("bounds"),
                "resolution": a.get("resolution"),
                "shape": a.get("shape"),
                "bands": a.get("bands"),
                "quality": a.get("quality"),
                "parents": a.get("parents"),
            }
            for a in artifacts
        ],
        "diagnostics": [
            {
                "code": d.get("code"),
                "severity": d.get("severity"),
                "message": d.get("message"),
                "suggested_actions": d.get("suggested_actions", []),
            }
            for d in diagnostics
        ],
        "warnings": warnings,
    }

    # Surface provenance summary (e.g. zonal statistics values) to the LLM
    provenance = result.get("provenance", {})
    if isinstance(provenance, dict) and "summary" in provenance:
        summary["result_values"] = provenance["summary"]

    return json.dumps(summary, indent=2, ensure_ascii=False, default=str)


def format_final_summary(trace: list[ToolCallRecord]) -> str:
    """Produce a human-readable summary of the entire agent execution."""
    lines: list[str] = [
        "=" * 60,
        "GeoHarness Agent — Execution Trace",
        "=" * 60,
    ]
    for record in trace:
        lines.append(
            f"\nStep {record.step}: {record.tool_name} → {record.status}"
        )
        if record.diagnostic_codes:
            lines.append(f"  Diagnostics: {', '.join(record.diagnostic_codes)}")
        for artifact in record.artifacts:
            lines.append(f"  Artifact: {artifact.id} ({artifact.type})")
            if artifact.shape:
                lines.append(f"    shape={artifact.shape}, crs={artifact.crs}")
    return "\n".join(lines)


# ── helpers ────────────────────────────────────────────────────────────────────


def _parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(v) for v in value)
    return None


def _parse_resolution(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(float(v) for v in value)
    return None


def _parse_shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return None
