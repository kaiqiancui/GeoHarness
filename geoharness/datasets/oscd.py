from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from geoharness.schemas import GeoArtifact
from geoharness.store import ArtifactStore
from geoharness.tools.raster import raster_artifact


OSCD_IMAGES_URL = (
    "https://huggingface.co/datasets/hkristen/oscd/resolve/main/"
    "Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip?download=true"
)
OSCD_TEST_LABELS_URL = "https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download"


def extract_zip_once(zip_path: str | Path, output_dir: str | Path) -> None:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    marker = output_dir / f".extracted_{zip_path.stem}"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    marker.write_text(str(zip_path), encoding="utf-8")


def find_oscd_root(extracted_dir: str | Path, contains: str) -> Path:
    extracted_dir = Path(extracted_dir)
    candidates = [
        path
        for path in extracted_dir.rglob("*")
        if path.is_dir() and contains.lower() in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find OSCD directory containing {contains!r} under {extracted_dir}")
    return candidates[0]


def list_cities(images_root: str | Path) -> list[str]:
    root = Path(images_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def build_city_multispectral_geotiff(
    images_root: str | Path,
    city: str,
    date_index: int,
    output_path: str | Path,
    *,
    band_names: tuple[str, ...] = ("B02", "B03", "B04", "B08"),
    prefer_rect: bool = True,
) -> Path:
    """Stack selected OSCD city bands into one GeoTIFF.

    OSCD ships each band as an individual GeoTIFF under `imgs_1` and `imgs_2`.
    This helper builds a compact 4-band image compatible with the Measure MVP.
    """

    if date_index not in (1, 2):
        raise ValueError("date_index must be 1 or 2")
    city_root = Path(images_root) / city
    rect_dir = city_root / f"imgs_{date_index}_rect"
    city_dir = rect_dir if prefer_rect and rect_dir.exists() else city_root / f"imgs_{date_index}"
    if not city_dir.exists():
        raise FileNotFoundError(city_dir)

    band_paths = [_find_band_path(city_dir, band_name) for band_name in band_names]
    arrays = []
    profile = None
    descriptions = []
    for band_name, band_path in zip(band_names, band_paths):
        with rasterio.open(band_path) as dataset:
            arrays.append(dataset.read(1))
            if profile is None:
                profile = dataset.profile.copy()
            descriptions.append(_canonical_band_name(band_name))
    if profile is None:
        raise RuntimeError("No OSCD bands were loaded.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=len(arrays), dtype=arrays[0].dtype)
    with rasterio.open(output_path, "w", **profile) as dst:
        for index, array in enumerate(arrays, start=1):
            dst.write(array, index)
            dst.set_band_description(index, descriptions[index - 1])
    return output_path


def write_full_scene_aoi(raster_path: str | Path, output_path: str | Path) -> Path:
    with rasterio.open(raster_path) as dataset:
        left, bottom, right, top = dataset.bounds
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "full_scene"},
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aoi, indent=2), encoding="utf-8")
    return output_path


def load_change_label(store: ArtifactStore, labels_root: str | Path, city: str, artifact_id: str) -> GeoArtifact:
    label_path = _find_label_path(Path(labels_root), city)
    artifact = raster_artifact(
        artifact_id,
        label_path,
        provenance={"tool": "OSCDAdapter", "role": "hidden_change_label"},
        bands=["change_label"],
    )
    artifact.metadata["oracle_only"] = True
    store.add(artifact)
    return artifact


def ndvi_change_summary(
    before_path: str | Path,
    after_path: str | Path,
    label_path: str | Path,
) -> dict:
    with rasterio.open(before_path) as before_ds, rasterio.open(after_path) as after_ds:
        before_bands = list(before_ds.descriptions)
        after_bands = list(after_ds.descriptions)
        before_ndvi = _ndvi(before_ds, before_bands)
        after_ndvi = _ndvi(after_ds, after_bands)
    delta = after_ndvi - before_ndvi

    with rasterio.open(label_path) as label_ds:
        label = label_ds.read(1)
    if label.shape != delta.shape:
        label = np.array(Image.fromarray(label).resize(delta.shape[::-1], Image.Resampling.NEAREST))
    # OSCD change maps use 1 for unchanged pixels and 2 for changed pixels.
    changed = label == 2
    unchanged = ~changed
    return {
        "changed_pixels": int(changed.sum()),
        "unchanged_pixels": int(unchanged.sum()),
        "mean_delta_ndvi_changed": float(np.nanmean(delta[changed])) if changed.any() else None,
        "mean_delta_ndvi_unchanged": float(np.nanmean(delta[unchanged])) if unchanged.any() else None,
        "absolute_delta_ndvi_changed": float(np.nanmean(np.abs(delta[changed]))) if changed.any() else None,
        "absolute_delta_ndvi_unchanged": float(np.nanmean(np.abs(delta[unchanged]))) if unchanged.any() else None,
    }


def register_city_artifacts(
    store: ArtifactStore,
    before_path: str | Path,
    after_path: str | Path,
) -> tuple[GeoArtifact, GeoArtifact]:
    before = raster_artifact(
        "oscd_before",
        before_path,
        provenance={"tool": "OSCDAdapter", "date_index": 1},
    )
    after = raster_artifact(
        "oscd_after",
        after_path,
        provenance={"tool": "OSCDAdapter", "date_index": 2},
    )
    store.add(before)
    store.add(after)
    return before, after


def _find_band_path(city_dir: Path, band_name: str) -> Path:
    matches = sorted(city_dir.glob(f"*{band_name}.tif")) + sorted(city_dir.glob(f"*{band_name}.TIF"))
    if not matches:
        matches = sorted(city_dir.glob(f"*{band_name}*.tif")) + sorted(city_dir.glob(f"*{band_name}*.TIF"))
    if not matches:
        raise FileNotFoundError(f"Could not find band {band_name} under {city_dir}")
    return matches[0]


def _find_label_path(labels_root: Path, city: str) -> Path:
    matches = sorted(labels_root.glob(f"{city}/cm/*-cm.tif"))
    if not matches:
        matches = sorted(labels_root.glob(f"{city}/cm/*.tif"))
    if not matches:
        raise FileNotFoundError(f"Could not find label for city {city} under {labels_root}")
    return matches[0]


def _canonical_band_name(band_name: str) -> str:
    mapping = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
    return mapping.get(band_name, band_name.lower())


def _ndvi(dataset: rasterio.io.DatasetReader, bands: list[str | None]) -> np.ndarray:
    if "nir" not in bands or "red" not in bands:
        raise ValueError(f"Expected red and nir band descriptions, got {bands}")
    nir = dataset.read(bands.index("nir") + 1).astype("float32")
    red = dataset.read(bands.index("red") + 1).astype("float32")
    denominator = nir + red
    return np.divide(nir - red, denominator, out=np.zeros_like(nir), where=np.abs(denominator) > 1e-6)
