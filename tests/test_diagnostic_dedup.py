from __future__ import annotations

import json
from pathlib import Path

from geoharness.experiments.fixtures import copy_with_high_nodata
from geoharness.synthetic import write_synthetic_measure_fixture
from geoharness.tasks.measure import run_measure_workflow


def test_repeated_validation_diagnostics_are_deduplicated(tmp_path: Path) -> None:
    raster_path, aoi_path = write_synthetic_measure_fixture(tmp_path / "inputs")
    warning_raster_path = copy_with_high_nodata(raster_path, tmp_path / "inputs" / "high_nodata.tif")

    summary = run_measure_workflow(
        store_root=tmp_path / "store",
        raster_path=warning_raster_path,
        aoi_path=aoi_path,
    )

    low_valid_keys = [
        (diagnostic["artifact_id"], diagnostic["check_name"])
        for diagnostic in summary["diagnostics"]
        if diagnostic["code"] == "low_valid_pixel_ratio"
    ]
    assert len(low_valid_keys) == len(set(low_valid_keys))
    assert ("raw_scene", "validate_raster") in low_valid_keys
    assert ("clipped_scene", "validate_raster") in low_valid_keys

    jsonl_records = [
        json.loads(line)
        for line in Path(summary["diagnostics_path"]).read_text(encoding="utf-8").splitlines()
    ]
    jsonl_keys = [
        (diagnostic["artifact_id"], diagnostic["check_name"])
        for diagnostic in jsonl_records
        if diagnostic["code"] == "low_valid_pixel_ratio"
    ]
    assert len(jsonl_keys) == len(set(jsonl_keys))
