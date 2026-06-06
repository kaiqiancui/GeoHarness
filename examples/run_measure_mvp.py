from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.measure import run_measure_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local GeoHarness Measure MVP.")
    parser.add_argument("--workdir", default="runs/measure_mvp", help="Output directory for inputs and artifacts.")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs_dir = workdir / "inputs"
    store_root = workdir / "store"
    raster_path, aoi_path = write_synthetic_measure_fixture(inputs_dir)
    summary = run_measure_workflow(
        store_root=store_root,
        raster_path=raster_path,
        aoi_path=aoi_path,
    )
    summary_path = workdir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
