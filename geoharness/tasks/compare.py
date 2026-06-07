"""Compare workflow — before/after change analysis through GeoHarness artifact pipeline.

Workflow:
  before + after + AOI -> validate pair -> clip both -> compute indices -> delta -> stats -> report
"""

from __future__ import annotations

from pathlib import Path

from geoharness.schemas import Diagnostic, GeoSkillResult
from geoharness.store import ArtifactStore, deduplicate_diagnostics
from geoharness.tools.compare import change_statistics, compute_delta, validate_raster_pair
from geoharness.tools.raster import clip_by_aoi_artifact, compute_index, load_raster
from geoharness.tools.stats import write_measure_report
from geoharness.tools.vector import load_vector


def run_compare_workflow(
    *,
    store_root: str | Path,
    before_raster_path: str | Path,
    after_raster_path: str | Path,
    aoi_path: str | Path,
    index_name: str = "NDVI",
) -> dict:
    """Run a Compare workflow (before/after change analysis).

    Parameters
    ----------
    before_raster_path : str | Path
        Path to the "before" (pre-event) GeoTIFF.
    after_raster_path : str | Path
        Path to the "after" (post-event) GeoTIFF.
    aoi_path : str | Path
        Path to the AOI GeoJSON.
    index_name : str
        Spectral index to compare (default ``"NDVI"``).
    """
    store = ArtifactStore(store_root)
    results: list[GeoSkillResult] = []

    # Step 1: load before raster
    results.append(load_raster(store, "before_scene", before_raster_path))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 2: load after raster
    results.append(load_raster(store, "after_scene", after_raster_path))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 3: load AOI vector
    results.append(load_vector(store, "aoi_vector", aoi_path))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 4: validate pair consistency
    results.append(validate_raster_pair(store, "before_scene", "after_scene"))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 5: clip before
    results.append(clip_by_aoi_artifact(store, "before_scene", "aoi_vector", "before_clipped"))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 6: clip after
    results.append(clip_by_aoi_artifact(store, "after_scene", "aoi_vector", "after_clipped"))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 7: compute before index
    results.append(compute_index(store, "before_clipped", "before_index", index_name=index_name))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 8: compute after index
    results.append(compute_index(store, "after_clipped", "after_index", index_name=index_name))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 9: compute delta
    metric_name = f"delta_{index_name.lower()}"
    results.append(compute_delta(store, "before_index", "after_index", "delta_raster", metric_name=metric_name))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 10: change statistics
    results.append(change_statistics(store, "delta_raster", "change_statistics"))
    if results[-1].status == "failed":
        return _summary(store, results)

    # Step 11: write compare report
    results.append(
        write_measure_report(
            store,
            "change_statistics",
            "compare_report",
            title=f"Change Detection ({index_name}) Report",
        )
    )

    return _summary(store, results)


def _summary(store: ArtifactStore, results: list[GeoSkillResult]) -> dict:
    diagnostics: list[Diagnostic] = [
        diagnostic
        for result in results
        for diagnostic in result.diagnostics
    ]
    diagnostics = deduplicate_diagnostics(diagnostics)
    status = "failed" if any(r.status == "failed" for r in results) else "success"
    if status == "success" and any(r.status == "warning" for r in results):
        status = "warning"
    return {
        "status": status,
        "artifacts": [a.to_dict() for a in store.all()],
        "diagnostics": [d.to_dict() for d in diagnostics],
        "metadata_path": str(store.metadata_path),
        "diagnostics_path": str(store.diagnostics_path),
    }
