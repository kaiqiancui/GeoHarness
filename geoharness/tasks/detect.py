"""Detect workflow — extends GeoHarness from continuous measurement to categorical detection.

Workflow:
  raster + AOI -> clip -> index -> threshold_mask -> mask_stats -> report
"""

from __future__ import annotations

from pathlib import Path

from geoharness.schemas import Diagnostic, GeoSkillResult
from geoharness.store import ArtifactStore, deduplicate_diagnostics
from geoharness.tools.masks import mask_area_statistics, threshold_mask
from geoharness.tools.raster import clip_by_aoi_artifact, compute_index, load_raster
from geoharness.tools.stats import write_measure_report
from geoharness.tools.vector import load_vector
from geoharness.tools.vectorize import vectorize_mask


def run_detect_workflow(
    *,
    store_root: str | Path,
    raster_path: str | Path,
    aoi_path: str | Path,
    target: str = "vegetation",
    threshold: float = 0.3,
) -> dict:
    """Run a Detect workflow.

    Parameters
    ----------
    target : str
        One of ``"vegetation"`` (NDVI > threshold) or ``"water"`` (NDWI > threshold).
    threshold : float
        Index threshold for binary mask.
    """
    if target == "vegetation":
        index_name = "NDVI"
    elif target == "water":
        index_name = "NDWI"
    else:
        raise ValueError(f"Unsupported target: {target}. Use 'vegetation' or 'water'.")

    mask_name = f"{target}_mask"
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

    results.append(compute_index(store, "clipped_scene", "index_raster", index_name=index_name))
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(
        threshold_mask(
            store,
            "index_raster",
            mask_name,
            threshold=threshold,
            mode="greater",
            mask_name=target,
        )
    )
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(vectorize_mask(store, mask_name, f"{target}_regions"))
    # vectorize warning (e.g. empty_mask) is non-fatal — continue to stats

    results.append(mask_area_statistics(store, mask_name, f"{target}_statistics"))
    if results[-1].status == "failed":
        return _summary(store, results)

    results.append(
        write_measure_report(
            store,
            f"{target}_statistics",
            f"{target}_report",
            title=f"{target.title()} Detection ({index_name}) Report",
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
