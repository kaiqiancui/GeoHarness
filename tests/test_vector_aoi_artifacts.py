from __future__ import annotations

from pathlib import Path

from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.measure import run_measure_workflow


def test_measure_workflow_tracks_aoi_as_vector_parent(tmp_path: Path) -> None:
    raster_path, aoi_path = write_synthetic_measure_fixture(tmp_path / "inputs")

    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=raster_path,
        aoi_path=aoi_path,
    )

    assert summary["status"] == "success"
    artifacts = {artifact["id"]: artifact for artifact in summary["artifacts"]}

    assert artifacts["aoi_vector"]["type"] == "vector"
    assert artifacts["aoi_vector"]["path"] == str(aoi_path)
    assert artifacts["aoi_vector"]["bounds"] is not None
    assert artifacts["aoi_vector"]["metadata"]["geometry_count"] == 1
    assert set(artifacts["clipped_scene"]["parents"]) == {"raw_scene", "aoi_vector"}
