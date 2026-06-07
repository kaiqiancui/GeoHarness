"""Change Vector Analysis (CVA) — multi-band spectral change score.

CVA computes the Euclidean distance between all-band pixel vectors
before and after, producing a continuous change score suitable for
threshold-based change detection.
"""

from __future__ import annotations

import numpy as np
import rasterio

from geoharness.feedback import validate_raster_artifact
from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore
from geoharness.tools.raster import raster_artifact


def compute_cva_score(
    store: ArtifactStore,
    before_raster_id: str,
    after_raster_id: str,
    output_id: str,
) -> GeoSkillResult:
    """Compute per-pixel CVA change score (multi-band spectral Euclidean distance).

    ``cva_score = sqrt(sum((band_after_i - band_before_i)^2))``

    Both rasters must have the same band count and shape.
    """
    before_src = store.get(before_raster_id)
    after_src = store.get(after_raster_id)
    diagnostics = validate_raster_artifact(before_src)
    diagnostics.extend(validate_raster_artifact(after_src))

    output_path = store.artifact_path(output_id, ".tif")
    with rasterio.open(before_src.path) as b_ds, rasterio.open(after_src.path) as a_ds:
        b_count = b_ds.count
        a_count = a_ds.count
        if b_count != a_count:
            diagnostics.append(
                Diagnostic(
                    code="band_count_mismatch",
                    severity="fatal",
                    message=f"Band count mismatch: before={b_count}, after={a_count}",
                    artifact_id=after_raster_id,
                    check_name="compute_cva_score",
                )
            )
            store.record_diagnostics(diagnostics)
            return GeoSkillResult(status="failed", diagnostics=diagnostics)

        # Read all bands
        b_data = b_ds.read().astype("float32")
        a_data = a_ds.read().astype("float32")

        # CVA: Euclidean distance per-pixel across bands
        diff = a_data - b_data
        cva = np.sqrt(np.sum(diff ** 2, axis=0))

        # Mask pixels where any band is nodata in either raster
        b_nodata = b_ds.nodata
        a_nodata = a_ds.nodata
        for i in range(b_count):
            if b_nodata is not None and not np.isnan(b_nodata):
                cva[b_data[i] == b_nodata] = np.nan
            if a_nodata is not None and not np.isnan(a_nodata):
                cva[a_data[i] == a_nodata] = np.nan
        # Also mask inf/nan from computation
        cva[~np.isfinite(cva)] = np.nan

        profile = a_ds.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(cva, 1)

    valid_count = int(np.sum(np.isfinite(cva)))

    artifact = raster_artifact(
        output_id,
        output_path,
        parents=[before_raster_id, after_raster_id],
        provenance={
            "tool": "ComputeCVA",
            "before": before_raster_id,
            "after": after_raster_id,
            "band_count": b_count,
        },
        bands=["cva_score"],
    )
    artifact.quality["valid_pixels"] = valid_count
    store.add(artifact)
    store.record_diagnostics(diagnostics)

    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "ComputeCVA"},
    )
