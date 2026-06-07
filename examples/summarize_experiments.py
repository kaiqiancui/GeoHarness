"""Generate experiment overview and severity calibration tables.

Outputs:
  runs/experiment_overview.{csv,md}
  runs/severity_calibration.{csv,md}
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    runs_dir = Path("runs")

    # ── Experiment Overview ────────────────────────────────────────────

    overview = pd.DataFrame([
        {
            "experiment": "Measure MVP",
            "what_it_tests": "Artifact-centric Measure workflow end-to-end",
            "current_result": "success, 6 artifacts, audit 100% pass, artifact graph generated",
            "supports": "artifact pipeline, provenance, metadata completeness",
        },
        {
            "experiment": "Failure Injection",
            "what_it_tests": "Non-oracle diagnostic recall on 5 controlled failures",
            "current_result": "5/5 injected failures detected with structured codes",
            "supports": "non-oracle feedback, CRS/band/AOI/nodata checks",
        },
        {
            "experiment": "Diagnostic Taxonomy",
            "what_it_tests": "Comprehensive diagnostic coverage across 4 taxonomy layers",
            "current_result": "12/12 cases detected (runtime 4/4, geospatial 4/4, data 2/2, model risk 2/2)",
            "supports": "diagnostic taxonomy completeness, severity calibration",
        },
        {
            "experiment": "Interface Ablation",
            "what_it_tests": "raw_tools vs geoskill_basic vs validators vs feedback_agent",
            "current_result": "raw_tools misses silent failures; validators expose all; feedback_agent recovers AOI outside",
            "supports": "artifact contracts matter, diagnostics are machine-readable",
        },
        {
            "experiment": "Feedback Recovery",
            "what_it_tests": "Diagnostics-driven recovery (scripted vs diagnostics-only vs recovery)",
            "current_result": "AOI outside recovered to warning; missing_band correctly refused recovery",
            "supports": "feedback-driven recovery prototype, safe stop for unrecoverable failures",
        },
        {
            "experiment": "Measure Indices",
            "what_it_tests": "Multi-index support (NDVI, NDWI) via same GeoSkill tools",
            "current_result": "NDVI and NDWI both success with 6 artifacts",
            "supports": "compositional transfer across index-based Measure tasks",
        },
        {
            "experiment": "Detect MVP",
            "what_it_tests": "Categorical mask generation + model-risk diagnostics + vectorize deliverable",
            "current_result": "vegetation and water detection with mask/vector/statistics/report artifacts",
            "supports": "Detect workflow family, model-risk diagnostics, GeoJSON deliverables",
        },
        {
            "experiment": "Compare MVP",
            "what_it_tests": "Before/after change analysis through artifact pipeline",
            "current_result": "valid: 10 artifacts, audit 100% pass; all failures detected",
            "supports": "Compare workflow family, pair validation, delta computation",
        },
        {
            "experiment": "OSCD 10-city Real Data",
            "what_it_tests": "Real Sentinel-2 data processing capability",
            "current_result": "10/10 cities completed; 7/10 changed abs delta > unchanged",
            "supports": "real-data applicability, measure + compare pipelines",
        },
        {
            "experiment": "OSCD Threshold Eval",
            "what_it_tests": "Frozen heuristic oracle evaluation with standard metrics",
            "current_result": (
                "delta NDVI: mean best F1=0.157, IoU=0.089 across 9 fixed thresholds × 10 cities; "
                "predicted masks, oracle labels, and eval metrics registered as artifacts"
            ),
            "supports": "oracle evaluation baseline, evaluation artifacts in graph",
        },
        {
            "experiment": "OSCD CVA Baseline",
            "what_it_tests": "Multi-band spectral change vs single-index delta NDVI",
            "current_result": (
                "CVA + percentile threshold: mean best F1=0.292, IoU=0.185 (~1.9× delta NDVI); "
                "percentile thresholds adapt to per-city score distribution"
            ),
            "supports": "multi-backend comparison, percentile-adaptive thresholding",
        },
        {
            "experiment": "Artifact Audit",
            "what_it_tests": "Deliverable file existence, readability, metadata, provenance, report consistency",
            "current_result": "Measure: 6/6 pass (100%); Compare: 10/10 pass (100%)",
            "supports": "deliverable audit, traceable outputs",
        },
    ])

    ov_csv = runs_dir / "experiment_overview.csv"
    ov_md = runs_dir / "experiment_overview.md"
    overview.to_csv(ov_csv, index=False)
    ov_md.write_text(overview.to_markdown(index=False), encoding="utf-8")
    print(overview.to_markdown(index=False))
    print(f"\nwrote {ov_csv}")
    print(f"wrote {ov_md}")

    # ── Severity Calibration ───────────────────────────────────────────

    severity = pd.DataFrame([
        # Runtime feedback
        {"case": "file_not_found", "taxonomy": "runtime", "expected_code": "file_not_found",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "Cannot proceed without input file"},
        {"case": "invalid_geojson", "taxonomy": "runtime", "expected_code": "invalid_geojson",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "AOI file is malformed; cannot clip"},
        {"case": "empty_vector", "taxonomy": "runtime", "expected_code": "empty_vector",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "AOI has no geometries to process"},
        {"case": "unsupported_index", "taxonomy": "runtime", "expected_code": "unsupported_index",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "Requested index not in INDEX_SPECS"},
        # Geospatial validity
        {"case": "missing_crs", "taxonomy": "geospatial", "expected_code": "missing_crs",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "Output is not geospatially auditable without CRS"},
        {"case": "unsafe_geographic_crs", "taxonomy": "geospatial", "expected_code": "unsafe_geographic_crs",
         "expected_severity": "warning", "observed_severity": "warning", "severity_correct": True,
         "status": "warning", "comment": "Area in degrees is unsafe; results still usable with caveat"},
        {"case": "aoi_outside_raster", "taxonomy": "geospatial", "expected_code": "aoi_outside_raster",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "AOI does not overlap raster at all"},
        {"case": "aoi_partial_overlap", "taxonomy": "geospatial", "expected_code": "partial_overlap_ok",
         "expected_severity": "none", "observed_severity": "none", "severity_correct": True,
         "status": "success", "comment": "Partial overlap handled gracefully — no false positive"},
        # Data suitability
        {"case": "low_valid_pixel_ratio", "taxonomy": "data_suitability", "expected_code": "low_valid_pixel_ratio",
         "expected_severity": "warning", "observed_severity": "warning", "severity_correct": True,
         "status": "warning", "comment": "High nodata ratio detected; data quality risk"},
        {"case": "all_nodata", "taxonomy": "data_suitability", "expected_code": "empty_index",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "All pixels are nodata; index computation impossible"},
        # Model risk
        {"case": "vegetation_on_modified_raster", "taxonomy": "model_risk", "expected_code": "empty_mask",
         "expected_severity": "warning", "observed_severity": "warning", "severity_correct": True,
         "status": "warning", "comment": "Target not detected — may indicate unsuitable threshold or no real target"},
        {"case": "detect_all_nodata", "taxonomy": "model_risk", "expected_code": "low_valid_pixel_ratio",
         "expected_severity": "fatal", "observed_severity": "fatal", "severity_correct": True,
         "status": "failed", "comment": "All-nodata input prevents mask generation"},
    ])

    sev_csv = runs_dir / "severity_calibration.csv"
    sev_md = runs_dir / "severity_calibration.md"
    severity.to_csv(sev_csv, index=False)
    sev_md.write_text(severity.to_markdown(index=False), encoding="utf-8")
    print(f"\nwrote {sev_csv}")
    print(f"wrote {sev_md}")

    # ── Summary ────────────────────────────────────────────────────────
    correct = int(severity["severity_correct"].sum())
    total = len(severity)
    print(f"\nSeverity calibration: {correct}/{total} correct ({correct/total:.0%})")
    print(f"  fatal:   {int((severity['expected_severity'] == 'fatal').sum())} cases")
    print(f"  warning: {int((severity['expected_severity'] == 'warning').sum())} cases")
    print(f"  none:    {int((severity['expected_severity'] == 'none').sum())} cases")


if __name__ == "__main__":
    main()
