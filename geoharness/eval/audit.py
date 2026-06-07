from __future__ import annotations

import json
from pathlib import Path

import rasterio

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


def audit_artifacts(metadata_path: str | Path) -> dict:
    """Run deliverable audit on all artifacts recorded in metadata.json."""
    metadata_path = Path(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts_by_id = payload["artifacts"]
    rows = []
    for artifact_id, artifact in artifacts_by_id.items():
        rows.append(_audit_one(artifact, artifacts_by_id))
    summary = _summarize(rows)
    return {"rows": rows, "summary": summary}


def _audit_one(artifact: dict, artifacts_by_id: dict[str, dict]) -> dict:
    artifact_id = artifact["id"]
    atype = artifact.get("type", "unknown")
    path = artifact.get("path", "")

    file_exists = bool(path) and Path(path).exists()
    readable = _check_readable(atype, path) if file_exists else False

    metadata_fields = _check_metadata(atype, artifact)
    metadata_complete = len(metadata_fields) == 0

    has_provenance = bool(artifact.get("provenance"))
    parents_exist = all(
        parent in artifacts_by_id for parent in artifact.get("parents", [])
    )

    report_consistent = _check_report_consistency(atype, artifact, artifacts_by_id)

    return {
        "artifact_id": artifact_id,
        "type": atype,
        "file_exists": file_exists,
        "readable": readable,
        "metadata_complete": metadata_complete,
        "missing_metadata_fields": ",".join(metadata_fields),
        "has_provenance": has_provenance,
        "parents_exist": parents_exist,
        "report_consistent": report_consistent,
    }


def _check_readable(atype: str, path: str) -> bool:
    artifact_path = Path(path)
    try:
        if atype == "raster":
            with rasterio.open(artifact_path):
                return True
        elif atype == "vector":
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            return "features" in data or "type" in data
        elif atype == "table":
            if pd is not None:
                pd.read_csv(artifact_path)
            else:
                content = artifact_path.read_text(encoding="utf-8")
                return len(content.strip()) > 0
            return True
        elif atype in ("report", "json"):
            return len(artifact_path.read_text(encoding="utf-8").strip()) > 0
        else:
            return artifact_path.exists()
    except Exception:
        return False


def _check_metadata(atype: str, artifact: dict) -> list[str]:
    missing = []
    if atype == "raster":
        for field in ("crs", "bounds", "resolution", "shape", "bands"):
            if artifact.get(field) is None:
                missing.append(field)
    elif atype == "vector":
        if artifact.get("bounds") is None:
            missing.append("bounds")
    elif atype in ("table", "report", "json"):
        if not artifact.get("provenance"):
            missing.append("provenance")
        if not artifact.get("parents"):
            missing.append("parents")
    return missing


def _check_report_consistency(atype: str, artifact: dict, artifacts_by_id: dict[str, dict]) -> bool | None:
    if atype != "report":
        return None
    path = artifact.get("path", "")
    if not path or not Path(path).exists():
        return False
    try:
        content = Path(path).read_text(encoding="utf-8")
    except Exception:
        return False
    # check report references at least one parent artifact
    parents = artifact.get("parents", [])
    for parent_id in parents:
        if parent_id in content:
            return True
    # fallback: check it mentions "Source artifact" or "artifact"
    if "artifact" in content.lower():
        return True
    return False


def _summarize(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {}
    return {
        "artifact_count": total,
        "file_exists_rate": sum(1 for r in rows if r["file_exists"]) / total,
        "readable_rate": sum(1 for r in rows if r["readable"]) / total,
        "metadata_completeness_rate": sum(1 for r in rows if r["metadata_complete"]) / total,
        "provenance_completeness_rate": sum(1 for r in rows if r["has_provenance"]) / total,
        "parent_reference_validity_rate": sum(1 for r in rows if r["parents_exist"]) / total,
        "audit_pass_rate": sum(
            1 for r in rows
            if r["file_exists"] and r["readable"] and r["metadata_complete"] and r["has_provenance"] and r["parents_exist"]
        ) / total,
    }
