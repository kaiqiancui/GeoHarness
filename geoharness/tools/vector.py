from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geoharness.feedback import validate_vector_artifact
from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore


def load_vector(store: ArtifactStore, artifact_id: str, path: str | Path) -> GeoSkillResult:
    diagnostics: list[Diagnostic] = []
    path = Path(path)
    if not path.exists():
        diagnostics.append(
            Diagnostic(
                code="file_not_found",
                severity="fatal",
                message=f"Vector file does not exist: {path}",
                artifact_id=artifact_id,
                check_name="load_vector",
            )
        )
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    try:
        geojson = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        diagnostics.append(
            Diagnostic(
                code="invalid_geojson",
                severity="fatal",
                message=f"Vector file is not valid JSON: {exc}",
                artifact_id=artifact_id,
                check_name="load_vector",
            )
        )
        store.record_diagnostics(diagnostics)
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    geometries = geojson_geometries(geojson)
    if not geometries:
        diagnostics.append(
            Diagnostic(
                code="empty_vector",
                severity="fatal",
                message="Vector GeoJSON contains no geometries.",
                artifact_id=artifact_id,
                check_name="load_vector",
            )
        )
        store.record_diagnostics(diagnostics)
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    bounds = _bounds_for_geometries(geometries)
    if bounds is None:
        diagnostics.append(
            Diagnostic(
                code="empty_vector",
                severity="fatal",
                message="Vector GeoJSON contains no coordinate data.",
                artifact_id=artifact_id,
                check_name="load_vector",
            )
        )
        store.record_diagnostics(diagnostics)
        return GeoSkillResult(status="failed", diagnostics=diagnostics)

    artifact = GeoArtifact(
        id=artifact_id,
        type="vector",
        path=str(path),
        bounds=bounds,
        provenance={"tool": "LoadVector", "source_path": str(path)},
        metadata={
            "driver": "GeoJSON",
            "geometry_count": len(geometries),
            "geojson_type": geojson.get("type"),
        },
    )
    diagnostics.extend(validate_vector_artifact(artifact))
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "LoadVector"},
    )


def geojson_geometries(geojson: dict[str, Any]) -> list[dict[str, Any]]:
    geojson_type = geojson.get("type")
    if geojson_type == "FeatureCollection":
        return [
            feature["geometry"]
            for feature in geojson.get("features", [])
            if isinstance(feature, dict) and feature.get("geometry") is not None
        ]
    if geojson_type == "Feature":
        geometry = geojson.get("geometry")
        return [geometry] if isinstance(geometry, dict) else []
    if geojson_type == "GeometryCollection":
        return [
            geometry
            for geometry in geojson.get("geometries", [])
            if isinstance(geometry, dict)
        ]
    if geojson_type in {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    }:
        return [geojson]
    return []


def _bounds_for_geometries(
    geometries: list[dict[str, Any]],
) -> tuple[float, float, float, float] | None:
    coordinates: list[tuple[float, float]] = []
    for geometry in geometries:
        coordinates.extend(_coordinate_pairs(geometry.get("coordinates")))
    if not coordinates:
        return None
    xs = [coordinate[0] for coordinate in coordinates]
    ys = [coordinate[1] for coordinate in coordinates]
    return (min(xs), min(ys), max(xs), max(ys))


def _coordinate_pairs(value: Any) -> list[tuple[float, float]]:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        return [(float(value[0]), float(value[1]))]
    if isinstance(value, list):
        coordinates: list[tuple[float, float]] = []
        for item in value:
            coordinates.extend(_coordinate_pairs(item))
        return coordinates
    return []
