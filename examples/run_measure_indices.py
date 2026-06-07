"""Run Measure workflow with multiple spectral indices on the same synthetic raster.

Demonstrates limited compositional transfer: the same GeoSkill pattern works
across NDVI, NDWI, etc. without per-index hardcoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.measure import run_measure_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Measure workflow with multiple indices.")
    parser.add_argument("--workdir", default="runs/measure_indices", help="Output directory.")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs = workdir / "inputs"
    raster_path, aoi_path = write_synthetic_measure_fixture(inputs)

    # The synthetic raster has blue/green/red/nir bands, so NDWI (green/nir) works too.
    # NDBI and NBR need SWIR which the synthetic data doesn't have — skip those here.
    indices_to_run = ["NDVI", "NDWI"]

    rows = []
    for index_name in indices_to_run:
        store_root = workdir / "stores" / index_name.lower()
        result = run_measure_workflow(
            store_root=store_root,
            raster_path=raster_path,
            aoi_path=aoi_path,
            index_name=index_name,
        )
        rows.append({
            "index_name": index_name,
            "status": result["status"],
            "artifact_count": len(result["artifacts"]),
            "diagnostic_codes": ",".join(d["code"] for d in result["diagnostics"]),
        })

    frame = pd.DataFrame(rows)
    csv_path = workdir / "index_results.csv"
    md_path = workdir / "index_results.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")
    print(frame.to_markdown(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
