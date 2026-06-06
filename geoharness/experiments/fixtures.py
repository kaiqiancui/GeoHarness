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
