from __future__ import annotations

from pathlib import Path

import pandas as pd

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
