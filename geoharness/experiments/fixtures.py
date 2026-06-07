from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def copy_without_crs(source: str | Path, target: str | Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(crs=None)
        data = src.read()
        descriptions = list(src.descriptions)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(descriptions, start=1):
            if description:
                dst.set_band_description(index, description)
    return target


def copy_without_band(source: str | Path, target: str | Path, *, remove_description: str = "nir") -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        descriptions = list(src.descriptions)
        remove_index = descriptions.index(remove_description) if remove_description in descriptions else src.count - 1
        keep_indexes = [index for index in range(1, src.count + 1) if index != remove_index + 1]
        data = src.read(keep_indexes)
        profile = src.profile.copy()
        profile.update(count=len(keep_indexes))
        kept_descriptions = [descriptions[index - 1] for index in keep_indexes]
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(kept_descriptions, start=1):
            if description:
                dst.set_band_description(index, description)
    return target


def copy_with_high_nodata(source: str | Path, target: str | Path, *, valid_fraction: float = 0.25) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        data = src.read()
        descriptions = list(src.descriptions)
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile.update(nodata=nodata)
    cutoff = int(data.shape[1] * valid_fraction)
    data[:, cutoff:, :] = nodata
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(descriptions, start=1):
            if description:
                dst.set_band_description(index, description)
    return target


def write_aoi_geojson(bounds: tuple[float, float, float, float], target: str | Path, *, name: str = "aoi") -> Path:
    left, bottom, right, top = bounds
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [left, bottom],
                            [right, bottom],
                            [right, top],
                            [left, top],
                            [left, bottom],
                        ]
                    ],
                },
            }
        ],
    }
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def write_unsafe_geographic_raster(target: str | Path) -> tuple[Path, Path]:
    """Create a valid-looking lat/lon raster for projection-safety checks."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    height = 64
    width = 64
    y, x = np.mgrid[0:height, 0:width]
    blue = (0.1 + x / width * 0.05).astype("float32")
    green = (0.2 + y / height * 0.05).astype("float32")
    red = (0.25 + x / width * 0.1).astype("float32")
    nir = (0.45 + y / height * 0.1).astype("float32")
    data = np.stack([blue, green, red, nir])
    transform = from_origin(116.0, 40.0, 0.0001, 0.0001)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(["blue", "green", "red", "nir"], start=1):
            dst.set_band_description(index, description)
    aoi = write_aoi_geojson(tuple(rasterio.open(target).bounds), target.with_suffix(".geojson"), name="latlon_aoi")
    return target, aoi


# ── Extended fixtures for diagnostic taxonomy stress test ──────────────────


def write_invalid_geojson(target: str | Path) -> Path:
    """Write a file that is NOT valid JSON (triggers invalid_geojson)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("this is not valid JSON {{{", encoding="utf-8")
    return target


def write_empty_geojson(target: str | Path) -> Path:
    """Write a valid GeoJSON FeatureCollection with zero features (triggers empty_vector)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"type": "FeatureCollection", "features": []}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def copy_with_all_nodata(source: str | Path, target: str | Path) -> Path:
    """Copy raster with ALL pixels set to nodata (triggers extreme low_valid_pixel_ratio)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile.update(nodata=nodata)
        descriptions = list(src.descriptions)
        # Fill all bands with nodata
        data = np.full_like(src.read(), nodata)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(descriptions, start=1):
            if description:
                dst.set_band_description(index, description)
    return target


def write_aoi_partial_overlap(
    base_raster: str | Path,
    target: str | Path,
    *,
    overlap_fraction: float = 0.5,
) -> Path:
    """Create an AOI that only partially overlaps the raster bounds.

    The AOI is shifted so only *overlap_fraction* of it falls within the raster.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(base_raster) as ds:
        left, bottom, right, top = ds.bounds

    width = right - left
    height = top - bottom

    # Shift AOI so only `overlap_fraction` remains inside raster
    aoi_left = left + width * (1.0 - overlap_fraction)
    aoi_bottom = bottom - height * 0.3  # partly outside on the bottom
    aoi_right = aoi_left + width * 0.6
    aoi_top = top * 1.0  # stays within top bound

    return write_aoi_geojson(
        (aoi_left, aoi_bottom, aoi_right, aoi_top),
        target,
        name="partial_overlap",
    )


def copy_with_resolution(source: str | Path, target: str | Path, *, resolution: tuple[float, float] = (20.0, 20.0)) -> Path:
    """Copy raster with a modified resolution (for resolution_mismatch testing)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        data = src.read()
        profile = src.profile.copy()
        descriptions = list(src.descriptions)
        # Build a new transform with the new resolution
        new_transform = from_origin(
            src.bounds.left, src.bounds.top,
            resolution[0], abs(resolution[1]),
        )
        profile.update(transform=new_transform, width=src.width, height=src.height,
                       res=resolution)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(descriptions, start=1):
            if description:
                dst.set_band_description(index, description)
    return target
