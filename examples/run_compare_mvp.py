"""Run Compare MVP — before/after change detection on synthetic data.

Generates a synthetic before/after pair (with known NIR change to simulate
vegetation loss) and runs the full Compare workflow through the GeoHarness
artifact pipeline, then against failure-injected pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.experiments.fixtures import (  # noqa: E402
    copy_without_band,
    copy_without_crs,
    write_aoi_geojson,
    write_unsafe_geographic_raster,
)
from geoharness.synthetic import write_synthetic_measure_fixture  # noqa: E402
from geoharness.tasks.compare import run_compare_workflow  # noqa: E402


def _modify_nir_band(source: str | Path, target: str | Path, *, nir_factor: float = 0.7) -> Path:
    """Copy a GeoTIFF while scaling the NIR band to simulate vegetation change.

    nir_factor < 1.0 → vegetation loss (NDVI decreases)
    nir_factor > 1.0 → vegetation gain (NDVI increases)
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        descriptions = list(src.descriptions)
        nir_index = descriptions.index("nir") if "nir" in descriptions else src.count - 1
        data = src.read()
        # Scale NIR band but keep it within reasonable range
        data[nir_index] = (data[nir_index].astype("float32") * nir_factor).astype(data.dtype)
        profile = src.profile.copy()
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for idx, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Compare workflow MVP on synthetic data.")
    parser.add_argument("--workdir", default="runs/compare_mvp")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs = workdir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    # Generate base synthetic data
    base_raster, base_aoi = write_synthetic_measure_fixture(inputs / "base")

    # Create a modified "after" raster with reduced NIR (simulating vegetation loss)
    after_raster = _modify_nir_band(base_raster, inputs / "after" / "after_scene.tif", nir_factor=0.6)

    # ── Run valid case ──────────────────────────────────────────────────
    result = run_compare_workflow(
        store_root=workdir / "stores" / "valid",
        before_raster_path=base_raster,
        after_raster_path=after_raster,
        aoi_path=base_aoi,
        index_name="NDVI",
    )
    summary_path = workdir / "valid_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Run failure cases ───────────────────────────────────────────────
    # Prepare failure fixtures
    no_crs = copy_without_crs(base_raster, inputs / "missing_crs_before" / "before.tif")
    missing_band = copy_without_band(after_raster, inputs / "missing_band_after" / "after.tif", remove_description="nir")
    unsafe_raster_before, unsafe_aoi = write_unsafe_geographic_raster(inputs / "unsafe_projection" / "before.tif")
    unsafe_raster_after, _ = write_unsafe_geographic_raster(inputs / "unsafe_projection" / "after.tif")
    outside_aoi = write_aoi_geojson((900000, 900, 900100, 1000), inputs / "outside_aoi.geojson", name="outside")

    failure_cases = [
        ("valid", base_raster, after_raster, base_aoi, None),
        ("missing_crs_before", no_crs, after_raster, base_aoi, "missing_crs"),
        ("missing_band_after", base_raster, missing_band, base_aoi, "missing_band"),
        ("unsafe_projection_pair", unsafe_raster_before, unsafe_raster_after, unsafe_aoi, "unsafe_geographic_crs"),
        ("aoi_outside", base_raster, after_raster, outside_aoi, "aoi_outside_raster"),
    ]

    rows = []
    for case_name, before_p, after_p, aoi_p, expected_code in failure_cases:
        result = run_compare_workflow(
            store_root=workdir / "stores" / case_name,
            before_raster_path=before_p,
            after_raster_path=after_p,
            aoi_path=aoi_p,
            index_name="NDVI",
        )
        codes = [d["code"] for d in result["diagnostics"]]
        detected = expected_code in codes if expected_code else True
        rows.append({
            "case": case_name,
            "status": result["status"],
            "expected_code": expected_code or "",
            "diagnostic_codes": ",".join(codes),
            "detected": detected,
            "artifact_count": len(result["artifacts"]),
        })

    frame = pd.DataFrame(rows)
    csv_path = workdir / "compare_results.csv"
    md_path = workdir / "compare_results.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")
    print(frame.to_markdown(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
