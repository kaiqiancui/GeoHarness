from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.mask import mask

from geoharness.feedback import validate_index_range, validate_raster_artifact, validate_vector_artifact
from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore
from geoharness.tools.indices import INDEX_SPECS, index_bands
from geoharness.tools.vector import geojson_geometries


def raster_artifact(
    artifact_id: str,
    path: str | Path,
    *,
    parents: Sequence[str] = (),
    provenance: dict | None = None,
    bands: list[str] | None = None,
) -> GeoArtifact:
    with rasterio.open(path) as dataset:
        array_shape = (dataset.count, dataset.height, dataset.width)
        nodata = dataset.nodata
        valid_ratio = _valid_ratio(dataset.read(1), nodata)
        descriptions = list(dataset.descriptions)
        inferred_bands = [
            description if description else f"band_{i}"
            for i, description in enumerate(descriptions, start=1)
        ]
        return GeoArtifact(
            id=artifact_id,
            type="raster",
            path=str(path),
            crs=str(dataset.crs) if dataset.crs else None,
            bounds=tuple(dataset.bounds),
            resolution=tuple(float(v) for v in dataset.res),
            transform=tuple(dataset.transform),
            shape=array_shape,
            bands=bands or inferred_bands,
            nodata=nodata,
            parents=list(parents),
            provenance=provenance or {},
            quality={"valid_pixel_ratio_band1": valid_ratio},
            metadata={"driver": dataset.driver, "dtype": dataset.dtypes[0]},
        )


def load_raster(store: ArtifactStore, artifact_id: str, path: str | Path) -> GeoSkillResult:
    diagnostics: list[Diagnostic] = []
    path = Path(path)
    if not path.exists():
        diagnostics.append(
            Diagnostic(
                code="file_not_found",
                severity="fatal",
                message=f"Raster file does not exist: {path}",
                artifact_id=artifact_id,
                check_name="load_raster",
            )
        )
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    artifact = raster_artifact(
        artifact_id,
        path,
        provenance={"tool": "LoadRaster", "source_path": str(path)},
    )
    diagnostics.extend(validate_raster_artifact(artifact))
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "LoadRaster"},
    )


def clip_by_aoi(
    store: ArtifactStore,
    raster_id: str,
    aoi_geojson_path: str | Path,
    output_id: str,
) -> GeoSkillResult:
    return _clip_by_aoi(
        store,
        raster_id,
        aoi_geojson_path,
        output_id,
        parents=[raster_id],
        provenance={"tool": "ClipByAOI", "aoi": str(aoi_geojson_path)},
        diagnostic_artifact_id=raster_id,
    )


def clip_by_aoi_artifact(
    store: ArtifactStore,
    raster_id: str,
    aoi_id: str,
    output_id: str,
) -> GeoSkillResult:
    aoi = store.get(aoi_id)
    diagnostics = validate_vector_artifact(aoi)
    if any(diagnostic.severity == "fatal" for diagnostic in diagnostics):
        store.record_diagnostics(diagnostics)
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    return _clip_by_aoi(
        store,
        raster_id,
        aoi.path,
        output_id,
        parents=[raster_id, aoi_id],
        provenance={"tool": "ClipByAOI", "aoi": aoi_id, "aoi_path": aoi.path},
        diagnostic_artifact_id=aoi_id,
    )


def _clip_by_aoi(
    store: ArtifactStore,
    raster_id: str,
    aoi_geojson_path: str | Path,
    output_id: str,
    *,
    parents: list[str],
    provenance: dict,
    diagnostic_artifact_id: str,
) -> GeoSkillResult:
    source = store.get(raster_id)
    output_path = store.artifact_path(output_id, ".tif")
    diagnostics = validate_raster_artifact(source)

    with open(aoi_geojson_path, encoding="utf-8") as handle:
        geojson = json.load(handle)
    geometries = geojson_geometries(geojson)
    if not geometries:
        diagnostics.append(
            Diagnostic(
                code="empty_aoi",
                severity="fatal",
                message="AOI GeoJSON contains no polygon geometries.",
                artifact_id=diagnostic_artifact_id,
                check_name="clip_by_aoi",
            )
        )
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    with rasterio.open(source.path) as src:
        try:
            clipped, transform = mask(src, geometries, crop=True, nodata=src.nodata)
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    code="aoi_outside_raster",
                    severity="fatal",
                    message=str(exc),
                    artifact_id=diagnostic_artifact_id,
                    check_name="clip_by_aoi",
                    suggested_actions=["check_aoi_bounds", "select_overlapping_raster"],
                )
            )
            store.record_diagnostics(diagnostics)
            return GeoSkillResult(status="failed", diagnostics=diagnostics)
        profile = src.profile.copy()
        profile.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
            }
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)

    artifact = raster_artifact(
        output_id,
        output_path,
        parents=parents,
        provenance=provenance,
        bands=source.bands,
    )
    diagnostics.extend(validate_raster_artifact(artifact))
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "ClipByAOI", "input": raster_id},
    )


def compute_index(
    store: ArtifactStore,
    raster_id: str,
    output_id: str,
    *,
    index_name: str = "NDVI",
    band_names: tuple[str, str] | None = None,
) -> GeoSkillResult:
    """Compute a spectral index.

    If *band_names* is omitted, required bands are looked up from INDEX_SPECS.
    """
    source = store.get(raster_id)
    diagnostics = validate_raster_artifact(source)

    if band_names is None:
        spec = INDEX_SPECS.get(index_name)
        if spec is None:
            diagnostics.append(
                Diagnostic(
                    code="unsupported_index",
                    severity="fatal",
                    message=f"Unsupported index: {index_name}. Available: {list(INDEX_SPECS)}",
                    artifact_id=raster_id,
                    check_name="compute_index",
                )
            )
            store.record_diagnostics(diagnostics)
            return GeoSkillResult(status="failed", diagnostics=diagnostics)
        band_names = spec["bands"]

    bands = source.bands or []
    missing = [name for name in band_names if name not in bands]
    if missing:
        diagnostics.append(
            Diagnostic(
                code="missing_band",
                severity="fatal",
                message=f"Missing required bands for {index_name}: {missing}",
                artifact_id=raster_id,
                check_name="compute_index",
                measured_value=bands,
                threshold=list(band_names),
            )
        )
        store.record_diagnostics(diagnostics)
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    output_path = store.artifact_path(output_id, ".tif")
    with rasterio.open(source.path) as src:
        left_idx = bands.index(band_names[0]) + 1
        right_idx = bands.index(band_names[1]) + 1
        left = src.read(left_idx).astype("float32")
        right = src.read(right_idx).astype("float32")
        denominator = left + right
        index = np.divide(
            left - right,
            denominator,
            out=np.zeros_like(left, dtype="float32"),
            where=np.abs(denominator) > 1e-6,
        )
        if src.nodata is not None:
            invalid = (left == src.nodata) | (right == src.nodata)
            index[invalid] = np.nan
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(index, 1)

    artifact = raster_artifact(
        output_id,
        output_path,
        parents=[raster_id],
        provenance={
            "tool": "ComputeIndex",
            "index": index_name,
            "formula": INDEX_SPECS.get(index_name, {}).get("formula", "unknown"),
        },
        bands=[index_name.lower()],
    )
    diagnostics.extend(validate_index_range(artifact))
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "ComputeIndex", "input": raster_id},
    )


def _valid_ratio(array: np.ndarray, nodata: float | int | None) -> float:
    if nodata is None:
        valid = np.isfinite(array)
    elif np.isnan(nodata):
        valid = np.isfinite(array)
    else:
        valid = array != nodata
    return float(valid.sum() / valid.size)
