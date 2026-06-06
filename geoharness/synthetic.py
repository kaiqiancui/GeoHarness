from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_synthetic_measure_fixture(root: str | Path) -> tuple[Path, Path]:
    """Create a small multispectral GeoTIFF and AOI GeoJSON for the MVP."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raster_path = root / "synthetic_scene.tif"
    aoi_path = root / "synthetic_aoi.geojson"

    height = 96
    width = 96
    y, x = np.mgrid[0:height, 0:width]
    red = 0.18 + 0.22 * (x / width) + 0.02 * np.sin(y / 8)
    nir = 0.55 - 0.18 * (x / width) + 0.08 * np.cos(y / 12)
    green = 0.25 + 0.12 * (y / height)
    blue = 0.12 + 0.08 * (x / width)
    stack = np.stack([blue, green, red, nir]).astype("float32")

    transform = from_origin(500_000, 4_100_000, 10, 10)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:32650",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(raster_path, "w", **profile) as dataset:
        dataset.write(stack)
        dataset.set_band_description(1, "blue")
        dataset.set_band_description(2, "green")
        dataset.set_band_description(3, "red")
        dataset.set_band_description(4, "nir")

    left = 500_180
    right = 500_760
    top = 4_099_820
    bottom = 4_099_260
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "synthetic_aoi"},
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
    aoi_path.write_text(json.dumps(aoi, indent=2), encoding="utf-8")
    return raster_path, aoi_path
