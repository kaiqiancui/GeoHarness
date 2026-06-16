"""Crop/navigate tools for spatial exploration (Case 2: football field counting)."""

from __future__ import annotations

import numpy as np
import rasterio

from geoharness.feedback import validate_raster_artifact
from geoharness.schemas import Diagnostic, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore
from geoharness.tools.raster import raster_artifact


def crop_view(
    store: ArtifactStore,
    raster_id: str,
    output_id: str,
    *,
    x: int = 0,
    y: int = 0,
    width: int = 320,
    height: int = 320,
) -> GeoSkillResult:
    """Crop a rectangular sub-view from a raster for spatial navigation.

    Use this to zoom into a specific region, shift the viewport, or check whether
    targets at the edge of the current view extend beyond the boundary.

    Parameters
    ----------
    raster_id : str
        Artifact ID of the source raster.
    output_id : str
        Artifact ID for the cropped output raster.
    x, y : int
        Top-left pixel coordinate of the crop window (0-based).
    width, height : int
        Size of the crop window in pixels.
    """
    source = store.get(raster_id)
    diagnostics = validate_raster_artifact(source)

    output_path = store.artifact_path(output_id, ".tif")
    with rasterio.open(source.path) as src:
        # Clamp to valid range
        x = max(0, min(x, src.width - 1))
        y = max(0, min(y, src.height - 1))
        width = min(width, src.width - x)
        height = min(height, src.height - y)

        window = rasterio.windows.Window(x, y, width, height)
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=height,
            width=width,
            transform=transform,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            for i, desc in enumerate(src.descriptions or [], start=1):
                if desc:
                    dst.set_band_description(i, desc)

    edge_info = []
    if y == 0:
        edge_info.append("top_edge")
    if x == 0:
        edge_info.append("left_edge")
    if y + height >= src.height:
        edge_info.append("bottom_edge")
    if x + width >= src.width:
        edge_info.append("right_edge")

    artifact = raster_artifact(
        output_id,
        output_path,
        parents=[raster_id],
        provenance={
            "tool": "CropView",
            "source": raster_id,
            "window": {"x": x, "y": y, "width": width, "height": height},
            "source_shape": [src.height, src.width],
            "at_edge": edge_info,
        },
        bands=source.bands,
    )
    artifact.metadata["crop_window"] = {"x": x, "y": y, "width": width, "height": height}
    artifact.metadata["at_image_edge"] = edge_info
    store.add(artifact)
    store.record_diagnostics(diagnostics)

    result = GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={
            "tool": "CropView",
            "window": {"x": x, "y": y, "width": width, "height": height},
            "at_edge": edge_info,
            "source_shape": [src.height, src.width],
        },
    )
    return result
