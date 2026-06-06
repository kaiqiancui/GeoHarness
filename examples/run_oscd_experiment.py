from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.datasets.oscd import (  # noqa: E402
    build_city_multispectral_geotiff,
    extract_zip_once,
    find_oscd_root,
    load_change_label,
    ndvi_change_summary,
    register_city_artifacts,
    write_full_scene_aoi,
)
from geoharness.store import ArtifactStore  # noqa: E402
from geoharness.tasks.measure import run_measure_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small real-data OSCD experiment.")
    parser.add_argument("--city", default="brasilia")
    parser.add_argument("--raw-dir", default="data/oscd/raw")
    parser.add_argument("--extract-dir", default="data/oscd/extracted")
    parser.add_argument("--workdir", default="runs/oscd_brasilia")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    workdir = Path(args.workdir)
    for zip_path in sorted(raw_dir.glob("*.zip")):
        extract_zip_once(zip_path, extract_dir)

    images_root = find_oscd_root(extract_dir, "Images")
    labels_root = find_oscd_root(extract_dir, "Labels")
    inputs_dir = workdir / "inputs"
    before_path = build_city_multispectral_geotiff(
        images_root,
        args.city,
        1,
        inputs_dir / f"{args.city}_before_rect.tif",
        prefer_rect=True,
    )
    after_path = build_city_multispectral_geotiff(
        images_root,
        args.city,
        2,
        inputs_dir / f"{args.city}_after_rect.tif",
        prefer_rect=True,
    )
    measure_after_path = build_city_multispectral_geotiff(
        images_root,
        args.city,
        2,
        inputs_dir / f"{args.city}_after_georef.tif",
        prefer_rect=False,
    )
    aoi_path = write_full_scene_aoi(measure_after_path, inputs_dir / f"{args.city}_aoi.geojson")

    measure_summary = run_measure_workflow(
        store_root=workdir / "measure_store",
        raster_path=measure_after_path,
        aoi_path=aoi_path,
    )

    compare_store = ArtifactStore(workdir / "compare_store")
    before_artifact, after_artifact = register_city_artifacts(compare_store, before_path, after_path)
    label_artifact = load_change_label(compare_store, labels_root, args.city, "oscd_change_label")
    compare_metrics = ndvi_change_summary(before_artifact.path, after_artifact.path, label_artifact.path)

    result = {
        "dataset": "OSCD",
        "city": args.city,
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "measure_status": measure_summary["status"],
        "measure_artifacts": [item["id"] for item in measure_summary["artifacts"]],
        "compare_metrics": compare_metrics,
        "compare_metadata_path": str(compare_store.metadata_path),
    }
    result_path = workdir / "oscd_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
