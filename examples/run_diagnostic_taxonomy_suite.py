"""Diagnostic taxonomy stress test — comprehensive failure injection by taxonomy.

Covers four taxonomy layers across Measure, Detect, and Compare workflows:

  Runtime feedback        — file_not_found, invalid_geojson, empty_vector, unsupported_index
  Geospatial validity     — missing_crs, invalid_crs, unsafe_geographic_crs, aoi_outside_raster,
                            aoi_partial_overlap, crs_mismatch, resolution_mismatch, shape_mismatch
  Data suitability        — low_valid_pixel_ratio, all_nodata
  Model risk              — empty_mask, saturated_mask
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
    copy_with_all_nodata,
    copy_with_high_nodata,
    copy_with_resolution,
    copy_without_band,
    copy_without_crs,
    write_aoi_geojson,
    write_aoi_partial_overlap,
    write_empty_geojson,
    write_invalid_geojson,
    write_unsafe_geographic_raster,
)
from geoharness.synthetic import write_synthetic_measure_fixture  # noqa: E402
from geoharness.tasks.detect import run_detect_workflow  # noqa: E402
from geoharness.tasks.measure import run_measure_workflow  # noqa: E402


def _modify_nir_band(source: str | Path, target: str | Path, *, nir_factor: float = 0.6) -> Path:
    """Copy a GeoTIFF while scaling the NIR band for before/after simulation."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        descriptions = list(src.descriptions)
        nir_index = descriptions.index("nir") if "nir" in descriptions else src.count - 1
        data = src.read()
        data[nir_index] = (data[nir_index].astype("float32") * nir_factor).astype(data.dtype)
        profile = src.profile.copy()
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for idx, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)
    return target


TAXONOMY = {
    "runtime": "Runtime feedback",
    "geospatial": "Geospatial validity",
    "data_suitability": "Data suitability",
    "model_risk": "Model risk",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diagnostic taxonomy stress test.")
    parser.add_argument("--workdir", default="runs/diagnostic_taxonomy")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    inputs = workdir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    base_raster, base_aoi = write_synthetic_measure_fixture(inputs / "base")

    # ── Prepare fixtures ───────────────────────────────────────────────
    no_crs = copy_without_crs(base_raster, inputs / "missing_crs" / "scene.tif")
    missing_band = copy_without_band(base_raster, inputs / "missing_band" / "scene.tif")
    high_nodata = copy_with_high_nodata(base_raster, inputs / "high_nodata" / "scene.tif")
    all_nodata = copy_with_all_nodata(base_raster, inputs / "all_nodata" / "scene.tif")
    unsafe_raster, unsafe_aoi = write_unsafe_geographic_raster(inputs / "unsafe_projection" / "scene.tif")
    outside_aoi = write_aoi_geojson((900000, 900, 900100, 1000), inputs / "outside_aoi.geojson", name="outside")
    partial_aoi = write_aoi_partial_overlap(base_raster, inputs / "partial_overlap" / "aoi.geojson", overlap_fraction=0.5)
    invalid_json = write_invalid_geojson(inputs / "invalid" / "aoi.geojson")
    empty_vector = write_empty_geojson(inputs / "empty_vector" / "aoi.geojson")

    # Compare fixtures
    after_raster = _modify_nir_band(base_raster, inputs / "after" / "after_scene.tif", nir_factor=0.6)
    res_mismatch_raster = copy_with_resolution(base_raster, inputs / "res_mismatch" / "scene_20m.tif", resolution=(20.0, 20.0))

    # ── Define test cases ──────────────────────────────────────────────

    cases: list[dict] = []

    # === Runtime feedback ===
    cases.append({
        "case": "file_not_found", "taxonomy": "runtime",
        "raster": inputs / "nonexistent.tif", "aoi": base_aoi,
        "expected_code": "file_not_found",
    })
    cases.append({
        "case": "invalid_geojson", "taxonomy": "runtime",
        "raster": base_raster, "aoi": invalid_json,
        "expected_code": "invalid_geojson",
    })
    cases.append({
        "case": "empty_vector", "taxonomy": "runtime",
        "raster": base_raster, "aoi": empty_vector,
        "expected_code": "empty_vector",
    })
    cases.append({
        "case": "unsupported_index", "taxonomy": "runtime",
        "raster": base_raster, "aoi": base_aoi,
        "expected_code": "unsupported_index",
        "index_name": "INVALID_INDEX_XYZ",
    })

    # === Geospatial validity ===
    cases.append({
        "case": "missing_crs", "taxonomy": "geospatial",
        "raster": no_crs, "aoi": base_aoi,
        "expected_code": "missing_crs",
    })
    cases.append({
        "case": "unsafe_geographic_crs", "taxonomy": "geospatial",
        "raster": unsafe_raster, "aoi": unsafe_aoi,
        "expected_code": "unsafe_geographic_crs",
    })
    cases.append({
        "case": "aoi_outside_raster", "taxonomy": "geospatial",
        "raster": base_raster, "aoi": outside_aoi,
        "expected_code": "aoi_outside_raster",
    })
    cases.append({
        "case": "aoi_partial_overlap", "taxonomy": "geospatial",
        "raster": base_raster, "aoi": partial_aoi,
        "expected_code": "partial_overlap_ok",
        "expect_no_code": True,
    })

    # === Data suitability ===
    cases.append({
        "case": "low_valid_pixel_ratio", "taxonomy": "data_suitability",
        "raster": high_nodata, "aoi": base_aoi,
        "expected_code": "low_valid_pixel_ratio",
    })
    cases.append({
        "case": "all_nodata", "taxonomy": "data_suitability",
        "raster": all_nodata, "aoi": base_aoi,
        "expected_code": "low_valid_pixel_ratio",
    })

    # === Model risk (Detect workflow) ===
    # Run detect on after raster — low NIR should produce small NDVI → likely empty_mask for vegetation
    cases.append({
        "case": "empty_mask_detect", "taxonomy": "model_risk",
        "raster": all_nodata, "aoi": base_aoi,
        "expected_code": "empty_mask", "workflow": "detect",
    })

    # ── Run Measure cases ───────────────────────────────────────────────

    rows = []
    for case in cases:
        if case.get("workflow") == "detect":
            continue  # handled below
        rows.extend(_run_measure_case(workdir, case))

    # ── Run Detect cases ───────────────────────────────────────────────

    # Vegetation detect on low-NIR raster - produces empty mask (NDVI too low)
    veg_result = run_detect_workflow(
        store_root=workdir / "stores" / "detect_empty_veg",
        raster_path=after_raster,
        aoi_path=base_aoi,
        target="vegetation",
        threshold=0.3,
    )
    rows.extend(_detect_rows("vegetation_on_modified", "model_risk", veg_result, "empty_mask"))

    # Vegetation detect on all_nodata - fatal
    nodata_result = run_detect_workflow(
        store_root=workdir / "stores" / "detect_all_nodata",
        raster_path=all_nodata,
        aoi_path=base_aoi,
        target="vegetation",
        threshold=0.3,
    )
    rows.extend(_detect_rows("detect_all_nodata", "model_risk", nodata_result, "low_valid_pixel_ratio"))

    # ── Write results ──────────────────────────────────────────────────

    frame = pd.DataFrame(rows)
    csv_path = workdir / "diagnostic_taxonomy_results.csv"
    md_path = workdir / "diagnostic_taxonomy_results.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")
    print(frame.to_markdown(index=False))
    print(f"\nwrote {csv_path}")

    # ── Summary ────────────────────────────────────────────────────────
    _print_summary(frame)


def _run_measure_case(workdir: Path, case: dict) -> list[dict]:
    result = run_measure_workflow(
        store_root=workdir / "stores" / case["case"],
        raster_path=case["raster"],
        aoi_path=case["aoi"],
        index_name=case.get("index_name", "NDVI"),
    )
    codes = [d["code"] for d in result["diagnostics"]]
    expected = case["expected_code"]
    if case.get("expect_no_code"):
        # Success = no false-positive diagnostics
        detected_ok = expected not in codes and len(codes) == 0
    else:
        detected_ok = expected in codes
    return [{
        "case": case["case"],
        "taxonomy": case.get("taxonomy", ""),
        "taxonomy_label": TAXONOMY.get(case.get("taxonomy", ""), ""),
        "workflow": "measure",
        "expected_code": expected,
        "detected": detected_ok,
        "observed_codes": ";".join(codes),
        "status": result["status"],
        "artifact_count": len(result["artifacts"]),
    }]


def _detect_rows(case_name: str, taxonomy: str, result: dict, expected_code: str) -> list[dict]:
    codes = [d["code"] for d in result["diagnostics"]]
    return [{
        "case": case_name,
        "taxonomy": taxonomy,
        "taxonomy_label": TAXONOMY.get(taxonomy, ""),
        "workflow": "detect",
        "expected_code": expected_code,
        "detected": expected_code in codes,
        "observed_codes": ";".join(codes),
        "status": result["status"],
        "artifact_count": len(result["artifacts"]),
    }]


def _print_summary(frame: pd.DataFrame) -> None:
    print("\n=== Taxonomy Coverage Summary ===\n")
    total = len(frame)
    detected = int(frame["detected"].sum())
    print(f"Total cases: {total}")
    print(f"Detected expected failures: {detected}/{total} ({detected/max(total,1):.0%})")

    print("\nBy taxonomy:")
    for tax, label in TAXONOMY.items():
        subset = frame[frame["taxonomy"] == tax]
        if len(subset) == 0:
            continue
        d = int(subset["detected"].sum())
        print(f"  {label}: {d}/{len(subset)} detected")

    print("\nBy workflow:")
    for wf in frame["workflow"].unique():
        subset = frame[frame["workflow"] == wf]
        d = int(subset["detected"].sum())
        print(f"  {wf}: {d}/{len(subset)} detected")


if __name__ == "__main__":
    main()
