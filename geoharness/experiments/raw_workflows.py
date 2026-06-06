from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


def raw_measure_workflow(raster_path: str | Path, aoi_path: str | Path, output_dir: str | Path) -> dict:
    """Minimal raw workflow without artifact contracts or structured diagnostics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ndvi_path = output_dir / "raw_ndvi.tif"
    stats_path = output_dir / "raw_stats.csv"

    with open(aoi_path, encoding="utf-8") as handle:
        geojson = json.load(handle)
    geometries = [feature["geometry"] for feature in geojson.get("features", [])]

    with rasterio.open(raster_path) as src:
        clipped, transform = mask(src, geometries, crop=True, nodata=src.nodata)
        descriptions = list(src.descriptions)
        nir_index = descriptions.index("nir")
        red_index = descriptions.index("red")
        nir = clipped[nir_index].astype("float32")
        red = clipped[red_index].astype("float32")
        denominator = nir + red
        ndvi = np.divide(nir - red, denominator, out=np.zeros_like(nir), where=np.abs(denominator) > 1e-6)
        profile = src.profile.copy()
        profile.update(count=1, height=ndvi.shape[0], width=ndvi.shape[1], transform=transform, dtype="float32", nodata=np.nan)
        with rasterio.open(ndvi_path, "w", **profile) as dst:
            dst.write(ndvi, 1)

    valid = ndvi[np.isfinite(ndvi)]
    pd.DataFrame(
        [
            {
                "valid_pixels": int(valid.size),
                "mean": float(valid.mean()),
                "min": float(valid.min()),
                "max": float(valid.max()),
            }
        ]
    ).to_csv(stats_path, index=False)
    return {
        "status": "success",
        "outputs": [str(ndvi_path), str(stats_path)],
        "diagnostics": [],
    }
