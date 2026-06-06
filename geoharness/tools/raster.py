from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask

from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore


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
    source = store.get(raster_id)
    output_path = store.artifact_path(output_id, ".tif")
    diagnostics = validate_raster_artifact(source)

    with open(aoi_geojson_path, encoding="utf-8") as handle:
        geojson = json.load(handle)
    geometries = _geojson_geometries(geojson)
    if not geometries:
        diagnostics.append(
            Diagnostic(
                code="empty_aoi",
                severity="fatal",
                message="AOI GeoJSON contains no polygon geometries.",
                artifact_id=raster_id,
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
                    artifact_id=raster_id,
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
        parents=[raster_id],
        provenance={"tool": "ClipByAOI", "aoi": str(aoi_geojson_path)},
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
    band_names: tuple[str, str] = ("nir", "red"),
) -> GeoSkillResult:
    source = store.get(raster_id)
    diagnostics = validate_raster_artifact(source)
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
            "formula": f"({band_names[0]} - {band_names[1]}) / ({band_names[0]} + {band_names[1]})",
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


def validate_raster_artifact(artifact: GeoArtifact) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if artifact.crs is None:
        diagnostics.append(
            Diagnostic(
                code="missing_crs",
                severity="fatal",
                message="Raster artifact has no CRS.",
                artifact_id=artifact.id,
                check_name="validate_raster",
            )
        )
    if artifact.bounds is None:
        diagnostics.append(
            Diagnostic(
                code="missing_bounds",
                severity="fatal",
                message="Raster artifact has no spatial bounds.",
                artifact_id=artifact.id,
                check_name="validate_raster",
            )
        )
    if artifact.crs is not None:
        try:
            crs = CRS.from_string(artifact.crs)
            if crs.is_geographic:
                diagnostics.append(
                    Diagnostic(
                        code="unsafe_geographic_crs",
                        severity="warning",
                        message="Raster uses a geographic CRS; area measurements should be done in a projected CRS.",
                        artifact_id=artifact.id,
                        check_name="validate_projection_safety",
                        measured_value=artifact.crs,
                        threshold="projected CRS",
                        suggested_actions=["reproject_to_local_projected_crs", "avoid_area_statistics_in_degrees"],
                    )
                )
        except Exception:
            diagnostics.append(
                Diagnostic(
                    code="invalid_crs",
                    severity="fatal",
                    message=f"Raster CRS cannot be parsed: {artifact.crs}",
                    artifact_id=artifact.id,
                    check_name="validate_raster",
                    measured_value=artifact.crs,
                )
            )
    valid_ratio = artifact.quality.get("valid_pixel_ratio_band1")
    if valid_ratio is not None and valid_ratio < 0.7:
        diagnostics.append(
            Diagnostic(
                code="low_valid_pixel_ratio",
                severity="warning",
                message=f"Only {valid_ratio:.1%} of band 1 pixels are valid.",
                artifact_id=artifact.id,
                check_name="validate_raster",
                measured_value=valid_ratio,
                threshold=0.7,
                suggested_actions=["choose_less_cloudy_scene", "reduce_aoi_or_report_uncertainty"],
            )
        )
    return diagnostics


def validate_index_range(artifact: GeoArtifact) -> list[Diagnostic]:
    diagnostics = validate_raster_artifact(artifact)
    with rasterio.open(artifact.path) as dataset:
        data = dataset.read(1, masked=True)
        if data.count() == 0:
            diagnostics.append(
                Diagnostic(
                    code="empty_index",
                    severity="fatal",
                    message="Index raster contains no valid pixels.",
                    artifact_id=artifact.id,
                    check_name="validate_index_range",
                )
            )
            return diagnostics
        min_value = float(data.min())
        max_value = float(data.max())
    if min_value < -1.001 or max_value > 1.001:
        diagnostics.append(
            Diagnostic(
                code="index_out_of_range",
                severity="warning",
                message=f"Index values should be in [-1, 1], got [{min_value:.3f}, {max_value:.3f}].",
                artifact_id=artifact.id,
                check_name="validate_index_range",
                measured_value={"min": min_value, "max": max_value},
                threshold={"min": -1, "max": 1},
            )
        )
    return diagnostics


def _valid_ratio(array: np.ndarray, nodata: float | int | None) -> float:
    if nodata is None:
        valid = np.isfinite(array)
    elif np.isnan(nodata):
        valid = np.isfinite(array)
    else:
        valid = array != nodata
    return float(valid.sum() / valid.size)


def _geojson_geometries(geojson: dict) -> list[dict]:
    if geojson.get("type") == "FeatureCollection":
        return [
            feature["geometry"]
            for feature in geojson.get("features", [])
            if feature.get("geometry") is not None
        ]
    if geojson.get("type") == "Feature":
        geometry = geojson.get("geometry")
        return [geometry] if geometry else []
    if geojson.get("type") in {"Polygon", "MultiPolygon"}:
        return [geojson]
    return []
