from __future__ import annotations

import rasterio
from rasterio.crs import CRS

from geoharness.schemas import Diagnostic, GeoArtifact


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


def validate_vector_artifact(artifact: GeoArtifact) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if artifact.type != "vector":
        diagnostics.append(
            Diagnostic(
                code="invalid_artifact_type",
                severity="fatal",
                message=f"Expected vector artifact, got {artifact.type}.",
                artifact_id=artifact.id,
                check_name="validate_vector",
                measured_value=artifact.type,
            )
        )
    if artifact.bounds is None:
        diagnostics.append(
            Diagnostic(
                code="missing_bounds",
                severity="fatal",
                message="Vector artifact has no spatial bounds.",
                artifact_id=artifact.id,
                check_name="validate_vector",
            )
        )
    geometry_count = artifact.metadata.get("geometry_count")
    if not isinstance(geometry_count, int) or geometry_count < 1:
        diagnostics.append(
            Diagnostic(
                code="empty_vector",
                severity="fatal",
                message="Vector artifact contains no geometries.",
                artifact_id=artifact.id,
                check_name="validate_vector",
                measured_value=geometry_count,
            )
        )
    return diagnostics
