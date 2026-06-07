"""Mask generation and risk diagnostics for the Detect workflow family."""

from __future__ import annotations

import numpy as np
import rasterio

from geoharness.feedback import validate_raster_artifact
from geoharness.schemas import Diagnostic, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore
from geoharness.tools.raster import raster_artifact


def threshold_mask(
    store: ArtifactStore,
    index_raster_id: str,
    output_id: str,
    *,
    threshold: float = 0.3,
    mode: str = "greater",
    mask_name: str = "target",
) -> GeoSkillResult:
    """Binarize an index raster by threshold.

    Parameters
    ----------
    mode : {"greater", "less"}
        ``greater`` → pixels where index > threshold become 1.
        ``less`` → pixels where index < threshold become 1.
    """
    source = store.get(index_raster_id)
    diagnostics = validate_raster_artifact(source)

    output_path = store.artifact_path(output_id, ".tif")
    with rasterio.open(source.path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        valid = np.isfinite(data) if nodata is None or np.isnan(nodata) else (data != nodata)

        if mode == "greater":
            mask = np.where(valid, (data > threshold).astype("uint8"), 0)
        elif mode == "less":
            mask = np.where(valid, (data < threshold).astype("uint8"), 0)
        else:
            diagnostics.append(
                Diagnostic(
                    code="invalid_parameter",
                    severity="fatal",
                    message=f"Unknown threshold mode: {mode}",
                    artifact_id=index_raster_id,
                    check_name="threshold_mask",
                )
            )
            store.record_diagnostics(diagnostics)
            return GeoSkillResult(status="failed", diagnostics=diagnostics)

        profile = src.profile.copy()
        # Binary mask: 0 = background (valid class), NOT nodata.
        # No nodata value is set so that valid_pixel_ratio reflects all pixels.
        profile.update(count=1, dtype="uint8", nodata=None)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

    valid_count = int(np.sum(valid))
    positive_count = int(np.sum(mask))
    positive_ratio = positive_count / valid_count if valid_count > 0 else 0.0

    artifact = raster_artifact(
        output_id,
        output_path,
        parents=[index_raster_id],
        provenance={
            "tool": "ThresholdMask",
            "input": index_raster_id,
            "threshold": threshold,
            "mode": mode,
            "mask_name": mask_name,
        },
        bands=[mask_name],
    )
    artifact.quality["positive_pixel_ratio"] = positive_ratio
    artifact.quality["positive_pixels"] = positive_count
    artifact.quality["valid_pixels"] = valid_count
    artifact.metadata["artifact_role"] = "binary_mask"
    artifact.metadata["mask_encoding"] = {"background": 0, "positive": 1}

    diagnostics.extend(validate_mask_risk(artifact))
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "ThresholdMask", "input": index_raster_id},
    )


def validate_mask_risk(artifact: "GeoArtifact") -> list[Diagnostic]:
    """Produce model-risk diagnostics for a binary mask artifact."""
    diagnostics: list[Diagnostic] = []
    quality = artifact.quality or {}
    ratio = quality.get("positive_pixel_ratio", 0.0)
    valid = quality.get("valid_pixels", 0)
    positive = quality.get("positive_pixels", 0)

    if valid == 0:
        diagnostics.append(
            Diagnostic(
                code="mask_has_no_valid_pixels",
                severity="fatal",
                message="Mask raster has zero valid pixels.",
                artifact_id=artifact.id,
                check_name="validate_mask_risk",
            )
        )
        return diagnostics

    if ratio == 0.0:
        diagnostics.append(
            Diagnostic(
                code="empty_mask",
                severity="warning",
                message="Mask contains zero positive pixels — target not detected.",
                artifact_id=artifact.id,
                check_name="validate_mask_risk",
                measured_value=ratio,
                threshold=0.0,
            )
        )
    elif ratio > 0.95:
        diagnostics.append(
            Diagnostic(
                code="saturated_mask",
                severity="warning",
                message=f"Mask covers {ratio:.1%} of valid area — nearly saturated.",
                artifact_id=artifact.id,
                check_name="validate_mask_risk",
                measured_value=ratio,
                threshold=0.95,
            )
        )
    elif ratio < 0.01:
        diagnostics.append(
            Diagnostic(
                code="low_positive_mask_ratio",
                severity="warning",
                message=f"Mask covers only {ratio:.3%} of valid area.",
                artifact_id=artifact.id,
                check_name="validate_mask_risk",
                measured_value=ratio,
                threshold=0.01,
            )
        )
    return diagnostics


def mask_area_statistics(
    store: ArtifactStore,
    mask_raster_id: str,
    output_id: str,
) -> GeoSkillResult:
    """Compute area statistics for a binary mask.

    If the mask is in a projected CRS, pixel area is computed from resolution.
    If geographic CRS, a diagnostic warning is emitted.
    """
    source = store.get(mask_raster_id)
    diagnostics = validate_raster_artifact(source)

    with rasterio.open(source.path) as src:
        data = src.read(1).astype("uint8")
        valid = data > 0
        total_valid = int(np.sum(data >= 0))
        positive = int(np.sum(data == 1))
        ratio = positive / total_valid if total_valid > 0 else 0.0

        crs = src.crs
        res = src.res
        is_projected = crs is not None and crs.is_projected
        if is_projected:
            pixel_area_m2 = abs(res[0] * res[1])
            estimated_area_m2 = positive * pixel_area_m2
            area_unit = "m2"
        else:
            pixel_area_m2 = None
            estimated_area_m2 = None
            area_unit = "unsafe_degree_units"
            diagnostics.append(
                Diagnostic(
                    code="unsafe_geographic_crs",
                    severity="warning",
                    message="Area computed in geographic CRS — units are degrees, not meters.",
                    artifact_id=mask_raster_id,
                    check_name="mask_area_statistics",
                    suggested_actions=["reproject_to_projected_crs"],
                )
            )

    output_path = store.artifact_path(output_id, ".csv")
    rows_text = (
        "artifact_id,valid_pixels,positive_pixels,positive_pixel_ratio,"
        "pixel_area,estimated_area,area_unit\n"
        f"{mask_raster_id},{total_valid},{positive},{ratio:.6f},"
        f"{pixel_area_m2 or ''},{estimated_area_m2 or ''},{area_unit}\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rows_text, encoding="utf-8")

    from geoharness.schemas import GeoArtifact
    artifact = GeoArtifact(
        id=output_id,
        type="table",
        path=str(output_path),
        parents=[mask_raster_id],
        provenance={"tool": "MaskAreaStatistics", "input": mask_raster_id},
        quality={
            "positive_pixel_ratio": ratio,
            "positive_pixels": positive,
            "estimated_area": estimated_area_m2,
            "area_unit": area_unit,
        },
        metadata={"columns": ["artifact_id", "valid_pixels", "positive_pixels", "positive_pixel_ratio", "pixel_area", "estimated_area", "area_unit"], "rows": 1},
    )
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "MaskAreaStatistics", "input": mask_raster_id},
    )
