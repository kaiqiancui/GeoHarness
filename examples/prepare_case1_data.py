"""Download Sentinel-2 L2A scenes for Case 1 using COG partial reads.

Strategy: Sentinel-2 data on Planetary Computer is stored as Cloud Optimized
GeoTIFFs (COGs). Rasterio can open these URLs directly and read only the
portion covering our AOI via HTTP Range requests — no need to download
the full 100MB+ granule.

Target: Brasilia, 3 cloud-free scenes from 2020.
Bands: B02, B03, B04, B08 (10m) + B11 (20m, SWIR for NDBI).
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject

BRASILIA_BBOX = [-47.92, -15.82, -47.84, -15.74]
BANDS = ["B02", "B03", "B04", "B08", "B11"]
BAND_DESC = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir"}


def _proxy_opener():
    proxy = os.environ.get("GEHARNESS_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"https": proxy}))
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(*handlers, urllib.request.HTTPSHandler(context=ctx))


def _search_scenes(bbox: list[float], date_range: str, max_cloud: int = 15) -> list[dict]:
    opener = _proxy_opener()
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": date_range,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 50,
    }
    req = urllib.request.Request(
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=30) as resp:
        data = json.loads(resp.read())

    scenes = []
    for feat in data.get("features", []):
        props = feat["properties"]
        assets = feat.get("assets", {})
        if all(b in assets for b in BANDS):
            scenes.append({
                "date": props["datetime"][:10],
                "cloud": props.get("eo:cloud_cover", 99),
                "id": feat["id"],
                "assets": {b: assets[b]["href"] for b in BANDS},
            })
    return sorted(scenes, key=lambda s: s["date"])


def _read_band_cog(url: str, aoi_bounds: tuple, dst_crs: str = "EPSG:4326") -> np.ndarray | None:
    """Read only the AOI window from a COG via HTTP Range request.

    Typically downloads 1-2MB instead of 100MB+ for a full granule.
    """
    try:
        with rasterio.open(url) as src:
            # Transform AOI from lat/lon to raster CRS
            from rasterio.warp import transform_bounds
            raster_bounds = transform_bounds(dst_crs, src.crs, *aoi_bounds)
            try:
                window = window_from_bounds(*raster_bounds, transform=src.transform)
            except Exception:
                return None
            window = window.round_lengths().round_offsets()
            # Clamp to raster extent
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            if window.width < 10 or window.height < 10:
                return None
            data = src.read(1, window=window)
            return data
    except Exception as e:
        print(f"    COG read error: {e}")
        return None


def _build_ms_geotiff(band_urls: dict[str, str], aoi_bounds: tuple, output_path: Path) -> Path | None:
    """Read AOI windows from COG URLs and stack into a multispectral GeoTIFF."""
    print(f"  Reading bands via COG partial reads...")

    # First read B04 (10m red) to establish the reference grid
    with rasterio.open(band_urls["B04"]) as src:
        from rasterio.warp import transform_bounds
        raster_bounds = transform_bounds("EPSG:4326", src.crs, *aoi_bounds)
        window = window_from_bounds(*raster_bounds, transform=src.transform)
        window = window.round_lengths().round_offsets()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        ref_transform = src.window_transform(window)
        ref_h = int(window.height)
        ref_w = int(window.width)
        ref_crs = src.crs
        ref_profile = src.profile.copy()
        ref_profile.update(height=ref_h, width=ref_w, transform=ref_transform)

    # Now read all 10m bands using the SAME window computed from B04
    arrays = []
    for band_name in ["B02", "B03", "B04", "B08"]:
        with rasterio.open(band_urls[band_name]) as src:
            # Transform AOI to this band's CRS
            band_raster_bounds = transform_bounds("EPSG:4326", src.crs, *aoi_bounds)
            band_window = window_from_bounds(*band_raster_bounds, transform=src.transform)
            band_window = band_window.round_lengths().round_offsets()
            band_window = band_window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            data = src.read(1, window=band_window)
            arrays.append(data.astype("float32"))
            print(f"    {band_name}: {data.shape[0]}x{data.shape[1]} (10m)")

    # Read B11 (20m) and resample to 10m using rasterio's reproject
    with rasterio.open(band_urls["B11"]) as src:
        b11_raster_bounds = transform_bounds("EPSG:4326", src.crs, *aoi_bounds)
        b11_window = window_from_bounds(*b11_raster_bounds, transform=src.transform)
        b11_window = b11_window.round_lengths().round_offsets()
        b11_window = b11_window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        b11_20m = src.read(1, window=b11_window).astype("float32")
        print(f"    B11: {b11_20m.shape[0]}x{b11_20m.shape[1]} (20m)")

        # Compute destination transform for 10m grid
        b11_window_bounds = src.window_bounds(b11_window)
        dst_transform = from_bounds(*b11_window_bounds, ref_w, ref_h)

    b11_10m = np.zeros((ref_h, ref_w), dtype="float32")
    reproject(
        source=b11_20m,
        destination=b11_10m,
        src_transform=src.window_transform(b11_window),
        src_crs=ref_crs,
        dst_transform=dst_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )
    arrays.append(b11_10m)

    # Use common min shape
    min_h = min(a.shape[0] for a in arrays)
    min_w = min(a.shape[1] for a in arrays)
    arrays = [a[:min_h, :min_w] for a in arrays]

    ref_profile.update(
        count=len(arrays), dtype="float32", height=min_h, width=min_w,
        driver="GTiff", nodata=0.0, compress="lzw",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **ref_profile) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr, i)
            dst.set_band_description(i, BAND_DESC[["B02", "B03", "B04", "B08", "B11"][i - 1]])

    return output_path


def main() -> None:
    workdir = Path("data/case1_temporal")
    workdir.mkdir(parents=True, exist_ok=True)

    print("Searching Sentinel-2 scenes over Brasilia...")
    scenes = _search_scenes(BRASILIA_BBOX, "2020-01-01/2020-12-31", max_cloud=15)

    # Pick 3 best scenes spread across the year
    selected = []
    for month_range in [(1, 4), (5, 8), (9, 12)]:
        candidates = [s for s in scenes
                      if month_range[0] <= int(s["date"][5:7]) <= month_range[1]
                      and s not in selected]
        if candidates:
            selected.append(min(candidates, key=lambda s: s["cloud"]))

    if len(selected) < 3:
        remaining = [s for s in scenes if s not in selected]
        selected.extend(sorted(remaining, key=lambda s: s["cloud"])[:3 - len(selected)])

    print(f"\nSelected {len(selected)} scenes:")
    for s in selected:
        print(f"  {s['date']}  cloud={s['cloud']:.1f}%  ({s['id'][:30]}...)")

    # Sign URLs via Planetary Computer
    try:
        import planetary_computer as pc
        for s in selected:
            s["assets"] = {b: pc.sign(url) for b, url in s["assets"].items()}
    except Exception:
        pass  # URLs may already be accessible

    # Download AOI crops for each scene
    ms_files = []
    for s in selected:
        date_str = s["date"]
        ms_path = workdir / f"brasilia_{date_str}_ms.tif"
        if ms_path.exists():
            print(f"\nSkipping {date_str} (already exists)")
            ms_files.append(ms_path)
            continue

        print(f"\nProcessing {date_str} (cloud={s['cloud']:.1f}%)...")
        result = _build_ms_geotiff(s["assets"], tuple(BRASILIA_BBOX), ms_path)
        if result:
            size_mb = result.stat().st_size / (1024 * 1024)
            print(f"  → Built: {result.name} ({size_mb:.1f} MB)")
            ms_files.append(result)

    print(f"\nDone! {len(ms_files)} scenes ready in {workdir}/")
    for f in sorted(ms_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
