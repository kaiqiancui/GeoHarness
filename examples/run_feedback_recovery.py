"""Feedback recovery experiment — 4 cases × 3 settings.

Cases:  valid, high_nodata, aoi_outside, missing_band
Settings:
  scripted_no_feedback     — ScriptedAgent, no diagnostics visible
  feedback_diagnostics_only — FeedbackAgent, diagnostics visible, no recovery
  feedback_with_generic_recovery — FeedbackAgent, diagnostics + recovery enabled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from geoharness.agents.feedback_agent import run_feedback_agent
from geoharness.agents.scripted_agent import run_scripted_agent
from geoharness.experiments.fixtures import (
    copy_with_high_nodata,
    copy_without_band,
    write_aoi_geojson,
)
from geoharness.synthetic import write_synthetic_measure_fixture

CASES = ["valid", "high_nodata", "aoi_outside", "missing_band"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feedback recovery experiment.")
    parser.add_argument("--workdir", default="runs/feedback_recovery", help="Output directory.")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs = workdir / "inputs"
    base_raster, base_aoi = write_synthetic_measure_fixture(inputs / "base")

    # Prepare case variants
    high_nodata_raster = copy_with_high_nodata(base_raster, inputs / "high_nodata" / "scene.tif")
    missing_band_raster = copy_without_band(base_raster, inputs / "missing_band" / "scene.tif")
    outside_aoi = write_aoi_geojson((900000, 900, 900100, 1000), inputs / "outside_aoi.geojson", name="outside")

    case_inputs = {
        "valid": (base_raster, base_aoi),
        "high_nodata": (high_nodata_raster, base_aoi),
        "aoi_outside": (base_raster, outside_aoi),
        "missing_band": (missing_band_raster, base_aoi),
    }

    rows = []
    for case_name in CASES:
        raster, aoi = case_inputs[case_name]
        rows.extend(_run_case(workdir, case_name, raster, aoi))

    frame = pd.DataFrame(rows)
    csv_path = workdir / "recovery_results.csv"
    md_path = workdir / "recovery_results.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")
    print(frame.to_markdown(index=False))
    print(f"\nwrote {csv_path}")


def _run_case(workdir: Path, case_name: str, raster: Path, aoi: Path) -> list[dict]:
    rows = []

    # Setting 1: ScriptedAgent (no feedback)
    result = run_scripted_agent(
        store_root=workdir / "scripted" / case_name / "store",
        raster_path=raster,
        aoi_path=aoi,
    )
    rows.append(_row(case_name, "scripted_no_feedback", result, False, False))

    # Setting 2: FeedbackAgent, diagnostics only (no recovery)
    result = run_feedback_agent(
        store_root=workdir / "feedback_diag" / case_name / "store",
        raster_path=raster,
        aoi_path=aoi,
        diagnostics_visible=True,
        recovery_enabled=False,
    )
    rows.append(_row(case_name, "feedback_diagnostics_only", result, result.get("recovery_attempted", False), result.get("recovery_success", False)))

    # Setting 3: FeedbackAgent, diagnostics + generic recovery
    result = run_feedback_agent(
        store_root=workdir / "feedback_recovery" / case_name / "store",
        raster_path=raster,
        aoi_path=aoi,
        diagnostics_visible=True,
        recovery_enabled=True,
    )
    rows.append(_row(case_name, "feedback_with_generic_recovery", result, result.get("recovery_attempted", False), result.get("recovery_success", False)))

    return rows


def _row(case: str, setting: str, result: dict, recovery_attempted: bool, recovery_success: bool) -> dict:
    trace = result.get("trace", [])
    return {
        "case": case,
        "setting": setting,
        "initial_status": trace[0]["status"] if trace else result["status"],
        "final_status": result["status"],
        "diagnostic_codes": ",".join(set(
            code for entry in trace for code in entry.get("diagnostics", [])
        )),
        "recovery_attempted": recovery_attempted,
        "recovery_success": recovery_success,
        "tool_call_count": result.get("tool_call_count", len(trace)),
        "artifact_count": result.get("artifact_count", len(result.get("artifacts", []))),
        "warning_reported": result.get("warning_reported", False),
    }


if __name__ == "__main__":
    main()
