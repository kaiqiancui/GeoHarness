"""Raw-tool implementations for the Exp 1 ablation baseline.

These functions deliberately bypass the GeoHarness artifact store, provenance
tracking and diagnostic engine.  They work with bare file paths and return
plain dicts.  When something goes wrong the agent sees a raw Python exception
rather than a structured diagnostic code.

Used as **Setting A (bare tools)** in the ablation experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from geoharness.tools.indices import INDEX_SPECS


# ── Raw tool implementations ──────────────────────────────────────────────────

def raw_load_raster(path: str | Path) -> dict[str, Any]:
    """Return basic raster metadata without any GeoHarness registration."""
    path = Path(path)
    with rasterio.open(path) as ds:
        bands = [d if d else f"band_{i}" for i, d in enumerate(ds.descriptions, start=1)]
        data = ds.read(1, masked=True)
        valid = int(data.count())
        total = int(data.size)
        return {
            "path": str(path),
            "crs": str(ds.crs) if ds.crs else "NOT SET",
            "bounds": list(ds.bounds),
            "shape": [ds.count, ds.height, ds.width],
            "resolution": list(ds.res),
            "bands": bands,
            "valid_pixels": valid,
            "total_pixels": total,
            "valid_ratio": valid / total if total > 0 else 0.0,
        }


def raw_load_vector(path: str | Path) -> dict[str, Any]:
    """Return basic vector metadata."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    return {
        "path": str(path),
        "geometry_count": len(features),
        "type": data.get("type", "unknown"),
    }


def raw_clip_by_aoi(raster_path: str | Path, aoi_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Clip a raster by an AOI GeoJSON and save the result."""
    raster_path = Path(raster_path)
    aoi_path = Path(aoi_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(aoi_path, encoding="utf-8") as fh:
        geojson = json.load(fh)
    geometries = [feat["geometry"] for feat in geojson.get("features", [])]
    if not geometries:
        raise ValueError("AOI file has no polygon geometries")

    with rasterio.open(raster_path) as src:
        clipped, transform = mask(src, geometries, crop=True, nodata=src.nodata)
        profile = src.profile.copy()
        profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)

    return {
        "output_path": str(output_path),
        "clipped_shape": list(clipped.shape),
        "width": int(clipped.shape[2]),
        "height": int(clipped.shape[1]),
        "message": f"Clipped raster saved to {output_path}",
    }


def raw_compute_index(
    raster_path: str | Path,
    output_path: str | Path,
    *,
    index_name: str = "NDVI",
) -> dict[str, Any]:
    """Compute a spectral index raster and return basic statistics."""
    raster_path = Path(raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = INDEX_SPECS.get(index_name)
    if spec is None:
        raise ValueError(f"Unsupported index: {index_name}. Available: {list(INDEX_SPECS)}")

    band_names = spec["bands"]
    with rasterio.open(raster_path) as src:
        bands = [d if d else f"band_{i}" for i, d in enumerate(src.descriptions, start=1)]
        try:
            left_idx = bands.index(band_names[0]) + 1
            right_idx = bands.index(band_names[1]) + 1
        except ValueError as e:
            raise ValueError(f"Missing bands for {index_name} ({band_names[0]}, {band_names[1]}). Available: {bands}") from e

        left = src.read(left_idx).astype("float32")
        right = src.read(right_idx).astype("float32")
        denominator = left + right
        index_arr = np.divide(
            left - right, denominator,
            out=np.zeros_like(left, dtype="float32"),
            where=np.abs(denominator) > 1e-6,
        )
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(index_arr, 1)

    valid = index_arr[np.isfinite(index_arr)]
    return {
        "output_path": str(output_path),
        "index_name": index_name,
        "formula": spec["formula"],
        "valid_pixels": int(valid.size),
        "mean": float(valid.mean()) if valid.size > 0 else None,
        "min": float(valid.min()) if valid.size > 0 else None,
        "max": float(valid.max()) if valid.size > 0 else None,
        "median": float(np.median(valid)) if valid.size > 0 else None,
        "std": float(valid.std()) if valid.size > 0 else None,
    }


def raw_stats(raster_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Compute descriptive statistics for a single-band raster."""
    raster_path = Path(raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as ds:
        data = ds.read(1, masked=True)
        values = data.compressed().astype("float64")
        if values.size == 0:
            raise ValueError("Raster has no valid pixels")

    summary = {
        "valid_pixels": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }

    pd.DataFrame([{
        "artifact_id": str(raster_path),
        **summary,
    }]).to_csv(output_path, index=False)

    summary["output_path"] = str(output_path)
    return summary


# ── LLM tool definitions for raw tools ────────────────────────────────────────

RAW_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "raw_load_raster",
        "description": (
            "Load a GeoTIFF raster and return basic metadata (CRS, bounds, shape, "
            "bands, valid pixel ratio). No provenance tracking is performed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the GeoTIFF file."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "raw_load_vector",
        "description": (
            "Load a GeoJSON vector file and return the geometry count."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the GeoJSON file."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "raw_clip_by_aoi",
        "description": (
            "Clip a raster by an AOI GeoJSON and save the result to output_path. "
            "No CRS validation, no provenance tracking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_path": {"type": "string", "description": "Path to the source GeoTIFF."},
                "aoi_path": {"type": "string", "description": "Path to the AOI GeoJSON."},
                "output_path": {"type": "string", "description": "Path for the clipped output GeoTIFF."},
            },
            "required": ["raster_path", "aoi_path", "output_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "raw_compute_index",
        "description": (
            "Compute a spectral index (NDVI, NDWI, NDBI, NBR) from a multi-band raster "
            "and save the result. Returns index statistics (mean, min, max). "
            "No validation of required bands — missing bands will raise an error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_path": {"type": "string", "description": "Path to the source multi-band GeoTIFF."},
                "output_path": {"type": "string", "description": "Path for the output index GeoTIFF."},
                "index_name": {
                    "type": "string",
                    "enum": ["NDVI", "NDWI", "NDBI", "NBR"],
                    "description": "Spectral index to compute.",
                },
            },
            "required": ["raster_path", "output_path", "index_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "raw_stats",
        "description": (
            "Compute descriptive statistics (mean, min, max, median, std) for a "
            "single-band raster and save them to a CSV file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_path": {"type": "string", "description": "Path to the raster."},
                "output_path": {"type": "string", "description": "Path for the output CSV."},
            },
            "required": ["raster_path", "output_path"],
            "additionalProperties": False,
        },
    },
]

RAW_HANDLERS: dict[str, Any] = {
    "raw_load_raster": raw_load_raster,
    "raw_load_vector": raw_load_vector,
    "raw_clip_by_aoi": raw_clip_by_aoi,
    "raw_compute_index": raw_compute_index,
    "raw_stats": raw_stats,
}
