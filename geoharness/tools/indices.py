"""Spectral index specifications for the Measure workflow family.

Each entry records required bands, formula, output band name, and valid range.
"""

from __future__ import annotations

INDEX_SPECS: dict[str, dict] = {
    "NDVI": {
        "bands": ("nir", "red"),
        "formula": "(nir - red) / (nir + red)",
        "output_band": "ndvi",
        "valid_range": (-1.0, 1.0),
    },
    "NDWI": {
        "bands": ("green", "nir"),
        "formula": "(green - nir) / (green + nir)",
        "output_band": "ndwi",
        "valid_range": (-1.0, 1.0),
    },
    "NDBI": {
        "bands": ("swir", "nir"),
        "formula": "(swir - nir) / (swir + nir)",
        "output_band": "ndbi",
        "valid_range": (-1.0, 1.0),
    },
    "NBR": {
        "bands": ("nir", "swir"),
        "formula": "(nir - swir) / (nir + swir)",
        "output_band": "nbr",
        "valid_range": (-1.0, 1.0),
    },
}


def index_bands(index_name: str) -> tuple[str, str]:
    """Return the (band_a, band_b) pair required for *index_name*."""
    spec = INDEX_SPECS.get(index_name)
    if spec is None:
        raise KeyError(f"unsupported index: {index_name}. Available: {list(INDEX_SPECS)}")
    return spec["bands"]


def index_formula(index_name: str) -> str:
    spec = INDEX_SPECS.get(index_name)
    if spec is None:
        raise KeyError(f"unsupported index: {index_name}")
    return spec["formula"]


def supported_indices() -> list[str]:
    return sorted(INDEX_SPECS)
