from __future__ import annotations

from pathlib import Path

from geoharness.schemas import Diagnostic, GeoSkillResult
from geoharness.store import ArtifactStore
from geoharness.tools.raster import clip_by_aoi_artifact, compute_index, load_raster
from geoharness.tools.stats import write_measure_report, zonal_statistics
from geoharness.tools.vector import load_vector


def run_measure_workflow(
    *,
    store_root: str | Path,
    raster_path: str | Path,
    aoi_path: str | Path,
) -> dict:
    store = ArtifactStore(store_root)
    results: list[GeoSkillResult] = []
    results.append(load_raster(store, "raw_scene", raster_path))
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(load_vector(store, "aoi_vector", aoi_path))
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(clip_by_aoi_artifact(store, "raw_scene", "aoi_vector", "clipped_scene"))
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(compute_index(store, "clipped_scene", "ndvi_raster", index_name="NDVI"))
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(zonal_statistics(store, "ndvi_raster", "ndvi_statistics"))
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(write_measure_report(store, "ndvi_statistics", "measure_report", title="NDVI Measure Workflow"))
    return _summary(store, results)


def _summary(store: ArtifactStore, results: list[GeoSkillResult]) -> dict:
    diagnostics: list[Diagnostic] = [
        diagnostic
        for result in results
        for diagnostic in result.diagnostics
    ]
    status = "failed" if any(result.status == "failed" for result in results) else "success"
    if status == "success" and any(result.status == "warning" for result in results):
        status = "warning"
    return {
        "status": status,
        "artifacts": [artifact.to_dict() for artifact in store.all()],
        "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
        "metadata_path": str(store.metadata_path),
        "diagnostics_path": str(store.diagnostics_path),
    }
