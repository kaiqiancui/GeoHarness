from __future__ import annotations

from pathlib import Path

import pandas as pd

from geoharness.experiments.fixtures import (
    copy_with_high_nodata,
    copy_without_band,
    copy_without_crs,
    write_unsafe_geographic_raster,
)
from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.measure import run_measure_workflow


def test_measure_workflow_outputs_artifacts(tmp_path: Path) -> None:
    raster_path, aoi_path = write_synthetic_measure_fixture(tmp_path / "inputs")
    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=raster_path,
        aoi_path=aoi_path,
    )

    assert summary["status"] == "success"
    artifact_ids = {artifact["id"] for artifact in summary["artifacts"]}
    assert {"raw_scene", "clipped_scene", "ndvi_raster", "ndvi_statistics", "measure_report"} <= artifact_ids

    table = next(artifact for artifact in summary["artifacts"] if artifact["id"] == "ndvi_statistics")
    frame = pd.read_csv(table["path"])
    assert frame.loc[0, "valid_pixels"] > 0
    assert -1 <= frame.loc[0, "mean"] <= 1


def test_outside_aoi_is_diagnosed(tmp_path: Path) -> None:
    raster_path, _ = write_synthetic_measure_fixture(tmp_path / "inputs")
    bad_aoi_path = tmp_path / "inputs" / "outside.geojson"
    bad_aoi_path.write_text(
        """{
          "type": "FeatureCollection",
          "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
              "type": "Polygon",
              "coordinates": [[[900000, 1000], [900100, 1000], [900100, 900], [900000, 900], [900000, 1000]]]
            }
          }]
        }""",
        encoding="utf-8",
    )

    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=raster_path,
        aoi_path=bad_aoi_path,
    )

    assert summary["status"] == "failed"
    assert "aoi_outside_raster" in {diagnostic["code"] for diagnostic in summary["diagnostics"]}


def test_missing_crs_is_diagnosed_as_fatal(tmp_path: Path) -> None:
    raster_path, aoi_path = write_synthetic_measure_fixture(tmp_path / "inputs")
    bad_raster_path = copy_without_crs(raster_path, tmp_path / "inputs" / "missing_crs.tif")

    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=bad_raster_path,
        aoi_path=aoi_path,
    )

    assert summary["status"] == "failed"
    assert "missing_crs" in _diagnostic_codes(summary)


def test_missing_band_is_diagnosed_as_fatal(tmp_path: Path) -> None:
    raster_path, aoi_path = write_synthetic_measure_fixture(tmp_path / "inputs")
    bad_raster_path = copy_without_band(raster_path, tmp_path / "inputs" / "missing_band.tif")

    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=bad_raster_path,
        aoi_path=aoi_path,
    )

    assert summary["status"] == "failed"
    assert "missing_band" in _diagnostic_codes(summary)


def test_high_nodata_is_diagnosed_as_warning(tmp_path: Path) -> None:
    raster_path, aoi_path = write_synthetic_measure_fixture(tmp_path / "inputs")
    warning_raster_path = copy_with_high_nodata(raster_path, tmp_path / "inputs" / "high_nodata.tif")

    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=warning_raster_path,
        aoi_path=aoi_path,
    )

    assert summary["status"] == "warning"
    assert "low_valid_pixel_ratio" in _diagnostic_codes(summary)


def test_unsafe_geographic_crs_is_diagnosed_as_warning(tmp_path: Path) -> None:
    raster_path, aoi_path = write_unsafe_geographic_raster(tmp_path / "inputs" / "latlon_scene.tif")

    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=raster_path,
        aoi_path=aoi_path,
    )

    assert summary["status"] == "warning"
    assert "unsafe_geographic_crs" in _diagnostic_codes(summary)


def _diagnostic_codes(summary: dict) -> set[str]:
    return {diagnostic["code"] for diagnostic in summary["diagnostics"]}
