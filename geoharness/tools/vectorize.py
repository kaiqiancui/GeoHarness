"""Vectorize raster masks to GeoJSON polygon deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features

from geoharness.feedback import validate_raster_artifact
from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore


def vectorize_mask(
    store: ArtifactStore,
    mask_raster_id: str,
    output_id: str,
    *,
    min_area: float = 400.0,
    target_class: str = "detected",
) -> GeoSkillResult:
    """Convert a binary mask raster to a GeoJSON FeatureCollection.

    Pixels with value > 0 are polygonized.  Polygons whose geographic area
    (in CRS units²) is smaller than *min_area* are dropped.

    The output GeoJSON is registered as a vector artifact with provenance
    linking back to the source mask.

    Parameters
    ----------
    min_area : float
        Minimum polygon area in CRS units² (e.g. m² for projected CRS).
        Default 400 ≈ 4 pixels at 10 m resolution.
    target_class : str
        Semantic label stored in GeoJSON feature properties.
    """
    source = store.get(mask_raster_id)
    diagnostics = validate_raster_artifact(source)

    output_path = store.artifact_path(output_id, ".geojson")

    with rasterio.open(source.path) as ds:
        data = ds.read(1).astype("int32")
        transform = ds.transform
        crs = ds.crs

        valid_mask = data > 0
        total_positive = int(np.sum(valid_mask))

        if total_positive == 0:
            diagnostics.append(
                Diagnostic(
                    code="empty_mask",
                    severity="warning",
                    message="Mask has no positive pixels — output is an empty FeatureCollection.",
                    artifact_id=mask_raster_id,
                    check_name="vectorize_mask",
                )
            )
            # Write empty FeatureCollection
            empty_geojson = {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {"source_mask": mask_raster_id, "total_positive_pixels": 0},
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(empty_geojson, indent=2), encoding="utf-8")
        else:
            # Polygonize positive pixels
            results = (
                {"properties": {"class": target_class, "pixel_area": int(area)}, "geometry": geometry}
                for geometry, value in features.shapes(
                    data.astype("uint8"), mask=valid_mask, transform=transform, connectivity=8
                )
                # shapes returns (geometry, pixel_value); compute area from geometry
                if (area := _polygon_pixel_area(geometry)) >= min_area
            )

            geojson = {
                "type": "FeatureCollection",
                "features": list(results),
                "metadata": {
                    "source_mask": mask_raster_id,
                    "min_area": min_area,
                    "total_positive_pixels": total_positive,
                    "crs": str(crs) if crs else None,
                },
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    # Build vector artifact
    feature_count = len(json.loads(output_path.read_text(encoding="utf-8"))["features"])
    artifact = GeoArtifact(
        id=output_id,
        type="vector",
        path=str(output_path),
        crs=str(crs) if crs else None,
        bounds=source.bounds,
        parents=[mask_raster_id],
        provenance={
            "tool": "VectorizeMask",
            "input": mask_raster_id,
            "min_area": min_area,
            "target_class": target_class,
        },
        quality={
            "total_positive_pixels": total_positive,
            "polygon_count": feature_count,
            "min_area": min_area,
        },
        metadata={
            "driver": "GeoJSON",
            "geometry_count": feature_count,
            "target_class": target_class,
        },
    )
    store.add(artifact)
    store.record_diagnostics(diagnostics)

    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "VectorizeMask", "input": mask_raster_id},
    )


def _polygon_pixel_area(geometry: dict) -> float:
    """Estimate polygon area from its coordinates using the shoelace formula.

    Falls back to the number of coordinate pairs as a rough approximation.
    """
    coords = geometry.get("coordinates", [])
    if not coords:
        return 0.0
    # Navigate into the outer ring of Polygon/MultiPolygon
    if geometry["type"] == "Polygon":
        ring = coords[0]
    elif geometry["type"] == "MultiPolygon":
        ring = coords[0][0] if coords and coords[0] else []
    else:
        return 0.0
    if len(ring) < 3:
        return 0.0
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    # Shoelace
    area = 0.5 * abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(ring))))
    if area == 0:
        return float(len(ring))
    return area
