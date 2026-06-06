from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.datasets.oscd import (  # noqa: E402
    build_city_multispectral_geotiff,
    extract_zip_once,
    find_oscd_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OSCD visualization panels.")
    parser.add_argument("--city", default="brasilia")
    parser.add_argument("--raw-dir", default="data/oscd/raw")
    parser.add_argument("--extract-dir", default="data/oscd/extracted")
    parser.add_argument("--output-dir", default="runs/oscd_visualizations")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    for zip_path in sorted(raw_dir.glob("*.zip")):
        extract_zip_once(zip_path, extract_dir)

    images_root = find_oscd_root(extract_dir, "Images")
    labels_root = find_oscd_root(extract_dir, "Labels")
    output_dir = Path(args.output_dir) / args.city
    input_dir = output_dir / "inputs"
    before_path = build_city_multispectral_geotiff(images_root, args.city, 1, input_dir / "before_rect.tif", prefer_rect=True)
    after_path = build_city_multispectral_geotiff(images_root, args.city, 2, input_dir / "after_rect.tif", prefer_rect=True)
    label_path = _find_label(labels_root, args.city)

    before = _read_stack(before_path)
    after = _read_stack(after_path)
    before_rgb = _rgb(before)
    after_rgb = _rgb(after)
    before_ndvi = _ndvi(before)
    after_ndvi = _ndvi(after)
    delta_ndvi = after_ndvi - before_ndvi
    with rasterio.open(label_path) as ds:
        label = ds.read(1)
    if label.shape != delta_ndvi.shape:
        label = np.array(Image.fromarray(label).resize(delta_ndvi.shape[::-1], Image.Resampling.NEAREST))
    changed = label == 2

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_image(output_dir / "before_rgb.png", before_rgb, title=f"{args.city} before RGB")
    _save_image(output_dir / "after_rgb.png", after_rgb, title=f"{args.city} after RGB")
    _save_raster(output_dir / "after_ndvi.png", after_ndvi, title="After NDVI", cmap="YlGn", vmin=-0.5, vmax=0.8)
    _save_raster(output_dir / "delta_ndvi.png", delta_ndvi, title="Delta NDVI", cmap="RdBu", vmin=-0.3, vmax=0.3)
    _save_raster(output_dir / "change_label.png", changed.astype(float), title="OSCD change label", cmap="gray_r", vmin=0, vmax=1)
    _save_panel(output_dir / "panel.png", before_rgb, after_rgb, after_ndvi, delta_ndvi, changed, args.city)
    print(f"wrote visualizations to {output_dir}")


def _read_stack(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read().astype("float32")


def _rgb(stack: np.ndarray) -> np.ndarray:
    # Stack order is blue, green, red, nir.
    rgb = np.stack([stack[2], stack[1], stack[0]], axis=-1)
    low = np.nanpercentile(rgb, 2)
    high = np.nanpercentile(rgb, 98)
    return np.clip((rgb - low) / (high - low + 1e-6), 0, 1)


def _ndvi(stack: np.ndarray) -> np.ndarray:
    red = stack[2]
    nir = stack[3]
    return np.divide(nir - red, nir + red, out=np.zeros_like(nir), where=np.abs(nir + red) > 1e-6)


def _save_image(path: Path, image: np.ndarray, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_raster(path: Path, data: np.ndarray, *, title: str, cmap: str, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_panel(
    path: Path,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    after_ndvi: np.ndarray,
    delta_ndvi: np.ndarray,
    changed: np.ndarray,
    city: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=160)
    axes = axes.ravel()
    axes[0].imshow(before_rgb)
    axes[0].set_title("Before RGB")
    axes[1].imshow(after_rgb)
    axes[1].set_title("After RGB")
    axes[2].imshow(changed, cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("OSCD change label")
    im3 = axes[3].imshow(after_ndvi, cmap="YlGn", vmin=-0.5, vmax=0.8)
    axes[3].set_title("After NDVI")
    im4 = axes[4].imshow(delta_ndvi, cmap="RdBu", vmin=-0.3, vmax=0.3)
    axes[4].set_title("Delta NDVI")
    overlay = after_rgb.copy()
    overlay[changed, 0] = 1.0
    overlay[changed, 1] *= 0.35
    overlay[changed, 2] *= 0.35
    axes[5].imshow(overlay)
    axes[5].set_title("Change overlay")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    fig.suptitle(f"OSCD {city}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _find_label(labels_root: Path, city: str) -> Path:
    matches = sorted((labels_root / city / "cm").glob("*-cm.tif"))
    if not matches:
        raise FileNotFoundError(city)
    return matches[0]


if __name__ == "__main__":
    main()
