from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.experiments.fixtures import (  # noqa: E402
    copy_without_band,
    copy_without_crs,
    copy_with_high_nodata,
    write_aoi_geojson,
)
from geoharness.experiments.raw_workflows import raw_measure_workflow  # noqa: E402
from geoharness.synthetic import write_synthetic_measure_fixture  # noqa: E402
from geoharness.tasks.measure import run_measure_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw-vs-GeoSkill ablation on controlled Measure tasks.")
    parser.add_argument("--workdir", default="runs/ablation")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs_dir = workdir / "inputs"
    base_raster, base_aoi = write_synthetic_measure_fixture(inputs_dir / "base")

    no_crs = copy_without_crs(base_raster, inputs_dir / "missing_crs" / "scene.tif")
    missing_band = copy_without_band(base_raster, inputs_dir / "missing_band" / "scene.tif")
    high_nodata = copy_with_high_nodata(base_raster, inputs_dir / "high_nodata" / "scene.tif")

    outside_aoi = inputs_dir / "outside_aoi.geojson"
    write_aoi_geojson((900000, 900, 900100, 1000), outside_aoi, name="outside")

    cases = [
        {"case": "valid", "raster": base_raster, "aoi": base_aoi, "expected": None},
        {"case": "missing_crs", "raster": no_crs, "aoi": base_aoi, "expected": "missing_crs"},
        {"case": "missing_band", "raster": missing_band, "aoi": base_aoi, "expected": "missing_band"},
        {"case": "high_nodata", "raster": high_nodata, "aoi": base_aoi, "expected": "low_valid_pixel_ratio"},
        {"case": "aoi_outside", "raster": base_raster, "aoi": outside_aoi, "expected": "aoi_outside_raster"},
    ]
    rows = []
    for case in cases:
        rows.extend(run_case(workdir, case))

    frame = pd.DataFrame(rows)
    output_csv = workdir / "ablation_results.csv"
    output_md = workdir / "ablation_results.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_md.write_text(frame.to_markdown(index=False), encoding="utf-8")
    print(frame.to_markdown(index=False))
    print(f"\nwrote {output_csv}")


def run_case(workdir: Path, case: dict) -> list[dict]:
    rows = []
    rows.append(run_raw(workdir, case))
    rows.append(run_geoskill(workdir, case))
    rows.append(run_geoskill_validators(workdir, case))
    return rows


def run_raw(workdir: Path, case: dict) -> dict:
    try:
        result = raw_measure_workflow(case["raster"], case["aoi"], workdir / "raw" / case["case"])
        status = result["status"]
        diagnostics = []
    except Exception as exc:
        status = "failed"
        diagnostics = [type(exc).__name__]
    return _row(case, "raw_tools", status, diagnostics, artifact_count=None, provenance_rate=None)


def run_geoskill(workdir: Path, case: dict) -> dict:
    # Same GeoSkill workflow, but count only workflow status. Diagnostics are not credited.
    result = run_measure_workflow(
        store_root=workdir / "geoskill_basic" / case["case"],
        raster_path=case["raster"],
        aoi_path=case["aoi"],
    )
    return _row(
        case,
        "geoskill_basic",
        result["status"],
        [],
        artifact_count=len(result["artifacts"]),
        provenance_rate=_provenance_rate(result),
    )


def run_geoskill_validators(workdir: Path, case: dict) -> dict:
    result = run_measure_workflow(
        store_root=workdir / "geoskill_validators" / case["case"],
        raster_path=case["raster"],
        aoi_path=case["aoi"],
    )
    diagnostics = [item["code"] for item in result["diagnostics"]]
    return _row(
        case,
        "geoskill_validators",
        result["status"],
        diagnostics,
        artifact_count=len(result["artifacts"]),
        provenance_rate=_provenance_rate(result),
    )


def _row(
    case: dict,
    interface: str,
    status: str,
    diagnostics: list[str],
    artifact_count: int | None,
    provenance_rate: float | None,
) -> dict:
    expected = case["expected"]
    return {
        "case": case["case"],
        "interface": interface,
        "status": status,
        "expected_failure": expected or "",
        "detected_expected_failure": bool(expected and expected in diagnostics),
        "diagnostic_codes": ",".join(diagnostics),
        "artifact_count": artifact_count,
        "provenance_rate": provenance_rate,
    }


def _provenance_rate(result: dict) -> float:
    artifacts = result["artifacts"]
    if not artifacts:
        return 0.0
    return sum(bool(item.get("provenance")) for item in artifacts) / len(artifacts)


if __name__ == "__main__":
    main()
