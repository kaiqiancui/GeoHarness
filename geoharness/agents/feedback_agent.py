"""Rule-based FeedbackAgent — observes diagnostics and makes execution decisions.

This agent executes a fixed Measure plan step by step. After each step it reads
structured diagnostics from the GeoSkillResult and uses a rule table to decide:
continue, continue_with_warning, stop, or retry (with generic recovery).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio

from geoharness.schemas import Diagnostic, GeoSkillResult
from geoharness.store import ArtifactStore, deduplicate_diagnostics
from geoharness.tools.raster import clip_by_aoi_artifact, compute_index, load_raster
from geoharness.tools.stats import write_measure_report, zonal_statistics
from geoharness.tools.vector import load_vector

# ── Decision rules ──────────────────────────────────────────────────────────

# (diagnostic_code, recovery_setting) -> decision
# decision is one of: "stop", "continue", "continue_with_warning", "retry_full_scene_aoi"
FATAL_CODES = {"missing_crs", "missing_band", "empty_vector", "invalid_geojson"}
WARNING_CODES = {"low_valid_pixel_ratio", "unsafe_geographic_crs", "index_out_of_range", "empty_index"}
RECOVERABLE_CODES = {"aoi_outside_raster"}


def decide(
    diagnostics: list[Diagnostic],
    recovery_enabled: bool = False,
) -> dict:
    """Return a decision dict based on diagnostics.

    Priority: recoverable > unrecoverable-fatal > warning > continue.
    (Recoverable codes are checked first so that fatal-severity codes like
     aoi_outside_raster can still trigger recovery.)
    """
    codes = [d.code for d in diagnostics]
    severities = {d.code: d.severity for d in diagnostics}

    if recovery_enabled:
        for code in codes:
            if code in RECOVERABLE_CODES:
                return _decision("retry_full_scene_aoi", code, diagnostics, "AOI outside raster — retry with full-scene AOI")

    for code in codes:
        if code in FATAL_CODES or severities.get(code) == "fatal":
            return _decision("stop", code, diagnostics, "fatal error — cannot safely recover")

    for code in codes:
        if code in WARNING_CODES or severities.get(code) == "warning":
            return _decision("continue_with_warning", code, diagnostics, "data quality risk — continuing with warning")

    return _decision("continue", None, diagnostics, "no issues detected")


def _decision(action: str, trigger_code: str | None, diagnostics: list[Diagnostic], reason: str) -> dict:
    return {
        "action": action,
        "trigger_code": trigger_code,
        "diagnostic_codes": [d.code for d in diagnostics],
        "reason": reason,
    }


def _write_full_scene_aoi_from_raster(raster_path: Path, aoi_path: Path) -> Path:
    """Generate a full-scene AOI from raster bounds (generic recovery)."""
    with rasterio.open(raster_path) as ds:
        left, bottom, right, top = ds.bounds
    import json
    aoi = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "recovery_full_scene"},
            "geometry": {"type": "Polygon", "coordinates": [[[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]]},
        }],
    }
    aoi_path.parent.mkdir(parents=True, exist_ok=True)
    aoi_path.write_text(json.dumps(aoi, indent=2), encoding="utf-8")
    return aoi_path


# ── FeedbackAgent ────────────────────────────────────────────────────────────

def run_feedback_agent(
    *,
    store_root: str | Path,
    raster_path: str | Path,
    aoi_path: str | Path,
    recovery_enabled: bool = False,
    diagnostics_visible: bool = True,
) -> dict:
    """Execute Measure workflow step by step, observing diagnostics each step.

    Parameters
    ----------
    recovery_enabled : bool
        If True, the agent may retry with generic recovery (e.g. full-scene AOI).
    diagnostics_visible : bool
        If False, the agent ignores diagnostics and always continues (no-feedback baseline).
    """
    raster_path = Path(raster_path)
    aoi_path = Path(aoi_path)
    store = ArtifactStore(store_root)
    trace: list[dict] = []
    recovery_attempted = False
    recovery_success = False
    warning_reported = False

    def _record(step_name: str, result: GeoSkillResult, decision: dict) -> None:
        trace.append({
            "step": step_name,
            "status": result.status,
            "diagnostics": [d.code for d in result.diagnostics],
            "decision": decision["action"],
            "trigger_code": decision.get("trigger_code"),
            "reason": decision.get("reason"),
        })

    # Step 1: load raster
    result = load_raster(store, "raw_scene", raster_path)
    diagnostic_list = result.diagnostics if diagnostics_visible else []
    decision = decide(diagnostic_list, recovery_enabled)
    _record("load_raster", result, decision)
    if decision["action"] == "stop":
        return _summary(store, trace, recovery_attempted, recovery_success, warning_reported)
    warning_reported = warning_reported or decision["action"] == "continue_with_warning"

    # Step 2: load vector (AOI)
    result = load_vector(store, "aoi_vector", aoi_path)
    diagnostic_list = result.diagnostics if diagnostics_visible else []
    decision = decide(diagnostic_list, recovery_enabled)
    _record("load_vector", result, decision)
    if decision["action"] == "stop":
        return _summary(store, trace, recovery_attempted, recovery_success, warning_reported)
    warning_reported = warning_reported or decision["action"] == "continue_with_warning"

    # Step 3: clip by AOI
    result = clip_by_aoi_artifact(store, "raw_scene", "aoi_vector", "clipped_scene")
    diagnostic_list = result.diagnostics if diagnostics_visible else []
    decision = decide(diagnostic_list, recovery_enabled)
    _record("clip_by_aoi", result, decision)

    if decision["action"] == "retry_full_scene_aoi":
        recovery_attempted = True
        recovery_aoi = _write_full_scene_aoi_from_raster(raster_path, Path(store_root) / "recovery_aoi.geojson")
        load_vector(store, "recovery_aoi", recovery_aoi)
        result = clip_by_aoi_artifact(store, "raw_scene", "recovery_aoi", "clipped_scene_recovery")
        diagnostic_list = result.diagnostics if diagnostics_visible else []
        decision = decide(diagnostic_list, recovery_enabled)
        _record("clip_retry_full_scene", result, decision)
        recovery_success = result.status != "failed"
        if result.status == "failed":
            return _summary(store, trace, recovery_attempted, recovery_success, warning_reported)
        warning_reported = True

    if decision["action"] == "stop":
        return _summary(store, trace, recovery_attempted, recovery_success, warning_reported)
    warning_reported = warning_reported or decision["action"] == "continue_with_warning"

    # Step 4: compute NDVI
    clip_id = "clipped_scene_recovery" if recovery_attempted else "clipped_scene"
    result = compute_index(store, clip_id, "ndvi_raster", index_name="NDVI")
    diagnostic_list = result.diagnostics if diagnostics_visible else []
    decision = decide(diagnostic_list, recovery_enabled)
    _record("compute_index", result, decision)
    if decision["action"] == "stop":
        return _summary(store, trace, recovery_attempted, recovery_success, warning_reported)
    warning_reported = warning_reported or decision["action"] == "continue_with_warning"

    # Step 5: zonal statistics
    result = zonal_statistics(store, "ndvi_raster", "ndvi_statistics")
    diagnostic_list = result.diagnostics if diagnostics_visible else []
    decision = decide(diagnostic_list, recovery_enabled)
    _record("zonal_statistics", result, decision)
    if decision["action"] == "stop":
        return _summary(store, trace, recovery_attempted, recovery_success, warning_reported)
    warning_reported = warning_reported or decision["action"] == "continue_with_warning"

    # Step 6: write report
    result = write_measure_report(store, "ndvi_statistics", "measure_report", title="NDVI Measure Workflow")
    diagnostic_list = result.diagnostics if diagnostics_visible else []
    decision = decide(diagnostic_list, recovery_enabled)
    _record("write_report", result, decision)
    warning_reported = warning_reported or decision["action"] == "continue_with_warning"

    return _summary(store, trace, recovery_attempted, recovery_success, warning_reported)


def _summary(
    store: ArtifactStore,
    trace: list[dict],
    recovery_attempted: bool,
    recovery_success: bool,
    warning_reported: bool,
) -> dict:
    # If recovery succeeded, the initial failure is overridden
    if recovery_attempted and recovery_success:
        status = "warning"  # recovered but with fallback — still worth flagging
    else:
        status = "success"
        for entry in trace:
            if entry["decision"] == "stop":
                status = "failed"
                break
            if entry["status"] == "failed":
                status = "failed"
                break
    if status != "failed" and (warning_reported or any(e.get("decision") == "continue_with_warning" for e in trace)):
        status = "warning"

    return {
        "agent": "FeedbackAgent",
        "status": status,
        "artifacts": [a.to_dict() for a in store.all()],
        "trace": trace,
        "recovery_attempted": recovery_attempted,
        "recovery_success": recovery_success,
        "warning_reported": warning_reported,
        "tool_call_count": len(trace),
        "artifact_count": len(store.all()),
        "metadata_path": str(store.metadata_path),
    }
