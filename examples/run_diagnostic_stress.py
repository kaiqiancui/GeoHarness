from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.measure import run_measure_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple injected-failure diagnostic stress test.")
    parser.add_argument("--workdir", default="runs/diagnostic_stress", help="Output directory.")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    raster_path, _ = write_synthetic_measure_fixture(workdir / "inputs")
    bad_aoi_path = workdir / "inputs" / "outside_aoi.geojson"
    bad_aoi_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "outside_aoi"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [900000, 1000],
                                    [900100, 1000],
                                    [900100, 900],
                                    [900000, 900],
                                    [900000, 1000],
                                ]
                            ],
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = run_measure_workflow(
        store_root=workdir / "store",
        raster_path=raster_path,
        aoi_path=bad_aoi_path,
    )
    summary_path = workdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    observed_codes = [item["code"] for item in summary["diagnostics"]]
    result = {
        "status": summary["status"],
        "expected_failure": "aoi_outside_raster",
        "observed_codes": observed_codes,
        "detected": "aoi_outside_raster" in observed_codes,
        "summary_path": str(summary_path),
    }
    result_path = workdir / "stress_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
