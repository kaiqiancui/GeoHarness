"""Satellite data catalog query tools (Case 1: multi-temporal analysis)."""

from __future__ import annotations

import json
import os
import ssl
import urllib.request

from geoharness.schemas import GeoSkillResult
from geoharness.store import ArtifactStore


def list_scenes(
    store: ArtifactStore,
    bbox: list[float],
    start_date: str,
    end_date: str,
    *,
    max_cloud: int = 15,
) -> dict:
    """Query the Sentinel-2 catalog for available scenes over an area.

    Searches Microsoft Planetary Computer STAC API for Sentinel-2 L2A scenes
    within the given bounding box and date range.

    Parameters
    ----------
    bbox : list[float]
        [left, bottom, right, top] in WGS84 (EPSG:4326) degrees.
    start_date, end_date : str
        Date range in YYYY-MM-DD format.
    max_cloud : int
        Maximum cloud cover percentage (default 15).

    Returns
    -------
    dict with a ``scenes`` key containing a list of {date, cloud_cover, scene_id}.
    """
    proxy = os.environ.get("GEHARNESS_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"https": proxy}))
    ctx = ssl.create_default_context()
    opener = urllib.request.build_opener(*handlers, urllib.request.HTTPSHandler(context=ctx))

    search_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start_date}/{end_date}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 30,
    }

    try:
        req = urllib.request.Request(
            search_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=30) as resp:
            data = json.loads(resp.read())

        scenes = []
        for feat in data.get("features", []):
            props = feat["properties"]
            scenes.append({
                "date": props["datetime"][:10],
                "datetime": props["datetime"],
                "cloud_cover": round(props.get("eo:cloud_cover", 99), 1),
                "scene_id": feat["id"],
                "platform": props.get("platform", "sentinel-2"),
            })

        scenes.sort(key=lambda s: s["datetime"])
        return {
            "status": "success",
            "artifacts": [],
            "diagnostics": [],
            "provenance": {
                "tool": "ListScenes",
                "bbox": bbox,
                "date_range": f"{start_date}/{end_date}",
                "count": len(scenes),
                "summary": {"scenes": scenes},
            },
        }
    except Exception as exc:
        return {
            "status": "failed",
            "artifacts": [],
            "diagnostics": [{
                "code": "catalog_query_error",
                "severity": "fatal",
                "message": f"Failed to query Sentinel-2 catalog: {exc}",
            }],
            "provenance": {"tool": "ListScenes"},
        }
