from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.experiments.fixtures import (  # noqa: E402
    copy_without_band,
    copy_without_crs,
    copy_with_high_nodata,
    write_aoi_geojson,
    write_unsafe_geographic_raster,
)
from geoharness.synthetic import write_synthetic_measure_fixture  # noqa: E402
from geoharness.tasks.measure import run_measure_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled failure injection cases.")
    parser.add_argument("--workdir", default="runs/failure_injection")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs = workdir / "inputs"
    base_raster, base_aoi = write_synthetic_measure_fixture(inputs / "base")
    no_crs = copy_without_crs(base_raster, inputs / "missing_crs" / "scene.tif")
    missing_band = copy_without_band(base_raster, inputs / "missing_band" / "scene.tif")
    high_nodata = copy_with_high_nodata(base_raster, inputs / "high_nodata" / "scene.tif")
    unsafe_raster, unsafe_aoi = write_unsafe_geographic_raster(inputs / "unsafe_projection" / "scene.tif")
    outside_aoi = write_aoi_geojson((900000, 900, 900100, 1000), inputs / "outside_aoi.geojson", name="outside")

    cases = [
        ("missing_crs", no_crs, base_aoi, "missing_crs"),
        ("missing_band", missing_band, base_aoi, "missing_band"),
        ("high_nodata", high_nodata, base_aoi, "low_valid_pixel_ratio"),
        ("unsafe_projection", unsafe_raster, unsafe_aoi, "unsafe_geographic_crs"),
        ("aoi_outside", base_raster, outside_aoi, "aoi_outside_raster"),
    ]

    rows = []
    for case_name, raster_path, aoi_path, expected_code in cases:
        result = run_measure_workflow(
            store_root=workdir / "stores" / case_name,
            raster_path=raster_path,
            aoi_path=aoi_path,
        )
        codes = [item["code"] for item in result["diagnostics"]]
        rows.append(
            {
                "case": case_name,
                "status": result["status"],
                "expected_code": expected_code,
                "diagnostic_codes": ",".join(codes),
                "detected": expected_code in codes,
                "artifact_count": len(result["artifacts"]),
            }
        )

    frame = pd.DataFrame(rows)
    output_csv = workdir / "failure_injection_results.csv"
    output_md = workdir / "failure_injection_results.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_md.write_text(frame.to_markdown(index=False), encoding="utf-8")
    print(frame.to_markdown(index=False))
    print(f"\nwrote {output_csv}")


if __name__ == "__main__":
    main()
