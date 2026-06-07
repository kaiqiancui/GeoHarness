"""Run Detect MVP — vegetation and water detection on synthetic data.

Demonstrates that the same GeoHarness artifact pipeline works for
categorical mask generation, not just continuous index measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.detect import run_detect_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Detect workflow MVP.")
    parser.add_argument("--workdir", default="runs/detect_mvp", help="Output directory.")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs = workdir / "inputs"
    raster_path, aoi_path = write_synthetic_measure_fixture(inputs)

    runs = [
        {"target": "vegetation", "threshold": 0.3},
        {"target": "water", "threshold": 0.2},
    ]

    rows = []
    for run_cfg in runs:
        store_root = workdir / "stores" / run_cfg["target"]
        result = run_detect_workflow(
            store_root=store_root,
            raster_path=raster_path,
            aoi_path=aoi_path,
            target=run_cfg["target"],
            threshold=run_cfg["threshold"],
        )
        rows.append({
            "target": run_cfg["target"],
            "threshold": run_cfg["threshold"],
            "status": result["status"],
            "artifact_count": len(result["artifacts"]),
            "diagnostic_codes": ",".join(d["code"] for d in result["diagnostics"]),
        })

        # Save summary
        summary_path = store_root.parent / f"{run_cfg['target']}_summary.json"
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    frame = pd.DataFrame(rows)
    csv_path = workdir / "detect_results.csv"
    md_path = workdir / "detect_results.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")
    print(frame.to_markdown(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
