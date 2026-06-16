"""Download satellite imagery for Case 2 (football field counting) via Google Maps Static API.

Downloads a few views at different locations/zoom levels to simulate spatial navigation.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from PIL import Image

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
SIZE = 640
ZOOM = 19  # closer zoom for clearer field visibility


def _download(lat: float, lon: float, zoom: int, session) -> np.ndarray:
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}&size={SIZE}x{SIZE}&maptype=satellite&key={API_KEY}"
    )
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def _save_geotiff(data: np.ndarray, lat: float, lon: float, path: Path) -> None:
    """Save numpy RGB array as GeoTIFF with approximate georeference."""
    import rasterio
    from rasterio.transform import from_origin

    # Approximate: at zoom 19, 1 pixel ≈ 0.3m, so 640px ≈ 192m ≈ 0.0017 deg
    deg_per_pixel = 0.0017 / 640
    h, w = data.shape[:2]
    transform = from_origin(
        lon - w * deg_per_pixel / 2,
        lat + h * deg_per_pixel / 2,
        deg_per_pixel,
        deg_per_pixel,
    )
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 3,
        "dtype": "uint8", "crs": "EPSG:4326", "transform": transform,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(3):
            dst.write(data[:, :, i], i + 1)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "green")
        dst.set_band_description(3, "blue")


def main() -> None:
    import requests
    session = requests.Session()
    proxy = os.environ.get("GEHARNESS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
        print(f"Using proxy: {proxy}")

    workdir = Path("data/case2_football")
    workdir.mkdir(parents=True, exist_ok=True)

    # Scene 1: Qingdao Sports Center — zoomed in view (several football training fields)
    lat, lon = 36.1042, 120.4678
    print(f"Downloading full scene at ({lat}, {lon}) zoom={ZOOM}...")
    full = _download(lat, lon, ZOOM, session)
    _save_geotiff(full, lat, lon, workdir / "full_scene.tif")
    print(f"  Saved: {full.shape[0]}x{full.shape[1]} (full scene)")

    # Scene 2: Slightly shifted view (simulates "move view")
    print(f"Downloading shifted view...")
    shifted = _download(lat + 0.0015, lon + 0.0015, ZOOM, session)
    _save_geotiff(shifted, lat + 0.0015, lon + 0.0015, workdir / "shifted_view.tif")
    print(f"  Saved: {shifted.shape[0]}x{shifted.shape[1]} (shifted)")

    # Scene 3: Zoomed out (lower zoom = wider area)
    print(f"Downloading zoomed-out view (zoom={ZOOM-2})...")
    zoomed_out = _download(lat, lon, ZOOM - 2, session)
    _save_geotiff(zoomed_out, lat, lon, workdir / "zoomed_out.tif")
    print(f"  Saved: {zoomed_out.shape[0]}x{zoomed_out.shape[1]} (zoomed out)")

    print("\nAll Case 2 data downloaded.")
    print(f"Files in {workdir}:")
    for f in sorted(workdir.glob("*.tif")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
