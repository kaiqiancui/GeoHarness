"""Compare workflow tools — pair validation, delta computation, and change statistics.

These tools support the Compare workflow family (before/after analysis):
  validate_raster_pair → compute_delta → change_statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from geoharness.feedback import validate_raster_artifact
from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore


def validate_raster_pair(
    store: ArtifactStore,
    before_id: str,
    after_id: str,
) -> GeoSkillResult:
    """Validate that before/after rasters are compatible for comparison.

    Checks CRS, resolution, shape, bounds overlap, and band availability.
    """
    before = store.get(before_id)
    after = store.get(after_id)
    diagnostics: list[Diagnostic] = []

    diagnostics.extend(validate_raster_artifact(before))
    diagnostics.extend(validate_raster_artifact(after))

    # CRS match
    if before.crs and after.crs and before.crs != after.crs:
        diagnostics.append(
            Diagnostic(
                code="crs_mismatch",
                severity="fatal",
                message=f"CRS mismatch: before={before.crs}, after={after.crs}",
                artifact_id=after_id,
                check_name="validate_raster_pair",
                measured_value={"before": before.crs, "after": after.crs},
            )
        )

    # Resolution match
    if before.resolution and after.resolution:
        if not _tolerably_close(before.resolution, after.resolution):
            diagnostics.append(
                Diagnostic(
                    code="resolution_mismatch",
                    severity="warning",
                    message=f"Resolution mismatch: before={before.resolution}, after={after.resolution}",
                    artifact_id=after_id,
                    check_name="validate_raster_pair",
                    measured_value={"before": before.resolution, "after": after.resolution},
                )
            )

    # Shape compatibility
    if before.shape and after.shape:
        b_h, b_w = before.shape[-2], before.shape[-1]
        a_h, a_w = after.shape[-2], after.shape[-1]
        if (b_h, b_w) != (a_h, a_w):
            diagnostics.append(
                Diagnostic(
                    code="shape_mismatch",
                    severity="warning",
                    message=f"Shape mismatch: before={(b_h, b_w)}, after={(a_h, a_w)}",
                    artifact_id=after_id,
                    check_name="validate_raster_pair",
                    measured_value={"before": (b_h, b_w), "after": (a_h, a_w)},
                )
            )

    # Bounds overlap
    if before.bounds and after.bounds:
        if not _bounds_overlap(before.bounds, after.bounds):
            diagnostics.append(
                Diagnostic(
                    code="bounds_no_overlap",
                    severity="warning",
                    message="Before and after rasters have no spatial overlap.",
                    artifact_id=after_id,
                    check_name="validate_raster_pair",
                    measured_value={"before": before.bounds, "after": after.bounds},
                )
            )

    # Band compatibility
    if before.bands and after.bands:
        missing_in_after = [b for b in before.bands if b not in after.bands]
        missing_in_before = [b for b in after.bands if b not in before.bands]
        if missing_in_after or missing_in_before:
            diagnostics.append(
                Diagnostic(
                    code="band_set_mismatch",
                    severity="warning",
                    message=f"Band mismatch. Missing in after: {missing_in_after}. Missing in before: {missing_in_before}.",
                    artifact_id=after_id,
                    check_name="validate_raster_pair",
                )
            )

    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        diagnostics=diagnostics,
        provenance={"tool": "ValidateRasterPair", "before": before_id, "after": after_id},
    )


def compute_delta(
    store: ArtifactStore,
    before_index_id: str,
    after_index_id: str,
    output_id: str,
    *,
    metric_name: str = "delta",
) -> GeoSkillResult:
    """Compute pixel-wise delta between two index rasters: after - before.

    Parameters
    ----------
    metric_name : str
        Display name for the change metric (e.g. ``"delta_ndvi"``).
    """
    before_src = store.get(before_index_id)
    after_src = store.get(after_index_id)
    diagnostics: list[Diagnostic] = []

    diagnostics.extend(validate_raster_artifact(before_src))
    diagnostics.extend(validate_raster_artifact(after_src))

    with rasterio.open(before_src.path) as b_ds, rasterio.open(after_src.path) as a_ds:
        before_data = b_ds.read(1).astype("float32")
        after_data = a_ds.read(1).astype("float32")

        # Use the after profile as output profile (prefer after projection)
        profile = a_ds.profile.copy()
        profile.update(dtype="float32", nodata=np.nan)

        if before_data.shape != after_data.shape:
            diagnostics.append(
                Diagnostic(
                    code="shape_mismatch",
                    severity="fatal",
                    message=f"Cannot compute delta: shape mismatch before={before_data.shape}, after={after_data.shape}",
                    artifact_id=after_index_id,
                    check_name="compute_delta",
                )
            )
            store.record_diagnostics(diagnostics)
            return GeoSkillResult(status="failed", diagnostics=diagnostics)

        # Mask nodata in both
        b_nodata = b_ds.nodata
        a_nodata = a_ds.nodata
        b_mask = np.isfinite(before_data) if b_nodata is None or np.isnan(b_nodata) else (before_data != b_nodata)
        a_mask = np.isfinite(after_data) if a_nodata is None or np.isnan(a_nodata) else (after_data != a_nodata)
        combined_mask = b_mask & a_mask

        delta = np.where(combined_mask, after_data - before_data, np.nan)

    output_path = store.artifact_path(output_id, ".tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(delta, 1)

    artifact = GeoArtifact(
        id=output_id,
        type="raster",
        path=str(output_path),
        crs=after_src.crs,
        bounds=after_src.bounds,
        resolution=after_src.resolution,
        transform=after_src.transform,
        shape=after_src.shape,
        bands=[metric_name],
        nodata=np.nan,
        parents=[before_index_id, after_index_id],
        provenance={
            "tool": "ComputeDelta",
            "before": before_index_id,
            "after": after_index_id,
            "metric": metric_name,
        },
        quality={"valid_pixels": int(np.sum(combined_mask))},
    )
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "ComputeDelta", "metric": metric_name},
    )


def change_statistics(
    store: ArtifactStore,
    delta_id: str,
    output_id: str,
    *,
    large_change_threshold: float = 0.2,
) -> GeoSkillResult:
    """Compute descriptive statistics for a delta raster.

    Parameters
    ----------
    large_change_threshold : float
        Absolute delta above this value is counted as "large change".
    """
    source = store.get(delta_id)
    diagnostics = validate_raster_artifact(source)

    with rasterio.open(source.path) as ds:
        data = ds.read(1)
        valid = np.isfinite(data)
        values = data[valid].astype("float64")

        if values.size == 0:
            diagnostics.append(
                Diagnostic(
                    code="empty_delta_raster",
                    severity="fatal",
                    message="Delta raster has no valid pixels.",
                    artifact_id=delta_id,
                    check_name="change_statistics",
                )
            )
            store.record_diagnostics(diagnostics)
            return GeoSkillResult(status="failed", diagnostics=diagnostics)

        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
        large = int(np.sum(np.abs(values) > large_change_threshold))

    output_path = store.artifact_path(output_id, ".csv")
    frame = pd.DataFrame([
        {
            "artifact_id": delta_id,
            "valid_pixels": int(values.size),
            "mean_delta": float(np.mean(values)),
            "median_delta": float(np.median(values)),
            "min_delta": float(np.min(values)),
            "max_delta": float(np.max(values)),
            "std_delta": float(np.std(values)),
            "positive_change_pixels": positive,
            "negative_change_pixels": negative,
            "large_change_pixels": large,
            "large_change_threshold": large_change_threshold,
        }
    ])
    frame.to_csv(output_path, index=False)

    artifact = GeoArtifact(
        id=output_id,
        type="table",
        path=str(output_path),
        parents=[delta_id],
        provenance={"tool": "ChangeStatistics", "input": delta_id},
        metadata={"columns": list(frame.columns), "rows": len(frame)},
    )
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={
            "tool": "ChangeStatistics",
            "input": delta_id,
            "summary": {
                "valid_pixels": int(values.size),
                "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)),
                "min_delta": float(np.min(values)),
                "max_delta": float(np.max(values)),
                "std_delta": float(np.std(values)),
                "positive_change_pixels": positive,
                "negative_change_pixels": negative,
                "large_change_pixels": large,
                "large_change_threshold": large_change_threshold,
            },
        },
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _tolerably_close(a: tuple[float, float], b: tuple[float, float], rtol: float = 1e-4) -> bool:
    return abs(a[0] - b[0]) < abs(a[0]) * rtol and abs(a[1] - b[1]) < abs(a[1]) * rtol


def _bounds_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    left_a, bottom_a, right_a, top_a = a
    left_b, bottom_b, right_b, top_b = b
    return not (right_a < left_b or right_b < left_a or top_a < bottom_b or top_b < bottom_a)
