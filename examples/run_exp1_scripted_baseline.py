"""Exp 1 baseline: fixed-script Measure workflow (no LLM, no Agent).

Runs the deterministic measure_workflow on the same 5 tasks as Exp 1.
This is the "no agent, no harness diagnostics" baseline column.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from geoharness.experiments.fixtures import (
    copy_without_band,
    copy_without_crs,
    copy_with_high_nodata,
    write_aoi_geojson,
)
from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.measure import run_measure_workflow


def main():
    workdir = Path("runs/exp1_scripted")
    inputs_dir = workdir / "inputs"
    base_raster, base_aoi = write_synthetic_measure_fixture(inputs_dir / "base")

    no_crs = copy_without_crs(base_raster, inputs_dir / "missing_crs" / "scene.tif")
    missing_band = copy_without_band(base_raster, inputs_dir / "missing_band" / "scene.tif")
    high_nodata = copy_with_high_nodata(base_raster, inputs_dir / "high_nodata" / "scene.tif")
    outside_aoi = write_aoi_geojson((900000, 900, 900100, 1000), inputs_dir / "outside_aoi.geojson")

    tasks = {
        "normal": (base_raster, base_aoi),
        "missing_crs": (no_crs, base_aoi),
        "aoi_outside": (base_raster, outside_aoi),
        "missing_band": (missing_band, base_aoi),
        "high_nodata": (high_nodata, base_aoi),
    }

    expected_codes = {
        "normal": [], "missing_crs": ["missing_crs"],
        "aoi_outside": ["aoi_outside_raster"], "missing_band": ["missing_band"],
        "high_nodata": ["low_valid_pixel_ratio"],
    }

    rows = []
    for task_key, (raster, aoi) in tasks.items():
        for rep in range(1, 4):
            print(f"Scripted: {task_key} rep {rep}/3")
            result = run_measure_workflow(
                store_root=workdir / "stores" / task_key / f"rep_{rep}",
                raster_path=raster, aoi_path=aoi,
            )
            codes = [d["code"] for d in result["diagnostics"]]
            expected = expected_codes[task_key]
            rows.append({
                "task": task_key, "rep": rep,
                "status": result["status"],
                "completed": result["status"] != "failed",
                "has_error": len(expected) > 0,
                "diagnostic_hit": any(c in codes for c in expected) if expected else True,
                "artifact_count": len(result["artifacts"]),
                "observed_codes": ";".join(codes),
            })

    frame = pd.DataFrame(rows)
    agg = frame.groupby("task").agg(
        success_rate=("completed", "mean"),
        diagnostic_hit_rate=("diagnostic_hit", "mean"),
        avg_artifacts=("artifact_count", "mean"),
    ).round(3)

    detail_csv = workdir / "scripted_detail.csv"
    agg_csv = workdir / "scripted_aggregate.csv"
    frame.to_csv(detail_csv, index=False)
    agg.to_csv(agg_csv)
    print("\n=== Scripted Baseline Results ===")
    print(agg.to_markdown())
    print(f"\nDetail: {detail_csv}")
    print(f"Aggregate: {agg_csv}")


if __name__ == "__main__":
    main()
