"""VLM vision tools — analyse geospatial imagery through LLM vision APIs.

Supports OpenAI GPT-4o and Anthropic Claude vision backends.
Provides the ``analyze_scene`` tool for the agent to inspect RGB images.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image

from geoharness.schemas import GeoArtifact
from geoharness.store import ArtifactStore


# ── RGB preview generation ────────────────────────────────────────────────────


def render_rgb_preview(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    *,
    red_band: int = 3,
    green_band: int = 2,
    blue_band: int = 1,
    percentile_clip: float = 2.0,
) -> Path:
    """Render an RGB preview PNG from a multi-band GeoTIFF.

    Parameters
    ----------
    raster_path : str or Path
        Path to the GeoTIFF raster.
    output_path : str or Path, optional
        Output PNG path.  If ``None``, writes alongside *raster_path* with
        ``_rgb.png`` suffix.
    red_band, green_band, blue_band : int
        1-based band indices for R, G, B channels.
    percentile_clip : float
        Percentile for contrast stretch (2.0 means clip at 2nd and 98th
        percentiles).

    Returns
    -------
    Path
        Path to the written PNG file.
    """
    raster_path = Path(raster_path)
    if output_path is None:
        output_path = raster_path.with_suffix("").with_suffix("")  # strip .tif
        output_path = Path(str(output_path) + "_rgb.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as ds:
        r = ds.read(red_band).astype("float32")
        g = ds.read(green_band).astype("float32")
        b = ds.read(blue_band).astype("float32")

    for channel in (r, g, b):
        vmin = float(np.nanpercentile(channel, percentile_clip))
        vmax = float(np.nanpercentile(channel, 100.0 - percentile_clip))
        if vmax > vmin:
            channel[:] = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
        else:
            channel[:] = 0
        channel[np.isnan(channel)] = 0

    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (rgb * 255).astype("uint8")
    Image.fromarray(rgb_uint8).save(output_path)
    return output_path


# ── VLM backends ──────────────────────────────────────────────────────────────


class OpenAIVisionClient:
    """Vision client using OpenAI GPT-4o."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        import os
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set for vision")
        self._base_url = base_url

    def analyze(self, image_path: str | Path, question: str) -> str:
        import os
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        # Proxy support
        proxy = os.environ.get("GEHARNESS_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxy=proxy, timeout=120.0)

        client = OpenAI(**client_kwargs)

        image_b64 = _encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


class AnthropicVisionClient:
    """Vision client using Anthropic Claude."""

    def __init__(self, api_key: str | None = None) -> None:
        import os
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY not set for vision")

    def analyze(self, image_path: str | Path, question: str) -> str:
        import os
        import anthropic

        kwargs: dict[str, Any] = {"api_key": self._api_key}
        proxy = os.environ.get("GEHARNESS_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            import httpx
            kwargs["http_client"] = httpx.Client(proxy=proxy, timeout=120.0)

        client = anthropic.Anthropic(**kwargs)

        image_b64 = _encode_image(image_path)
        media_type = _guess_media_type(image_path)

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        return response.content[0].text if response.content else ""


class QwenVisionClient:
    """Vision client using Qwen2.5-VL via Alibaba DashScope (OpenAI-compatible)."""

    DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str | None = None, model: str = "qwen3-vl-plus") -> None:
        import os
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY not set for Qwen VL")
        self._model = model

    def analyze(self, image_path: str | Path, question: str) -> str:
        import os
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "base_url": self.DASHSCOPE_BASE,
        }
        proxy = os.environ.get("GEHARNESS_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxy=proxy, timeout=120.0)

        client = OpenAI(**client_kwargs)

        image_b64 = _encode_image(image_path)
        media_type = _guess_media_type(image_path)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


def _encode_image(image_path: str | Path) -> str:
    """Read an image file and return a base64-encoded string."""
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def _guess_media_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    mapping = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    return mapping.get(suffix, "image/png")


# ── GeoHarness tool: analyze_scene ────────────────────────────────────────────


_vision_client: Any = None


def _get_vision_client() -> Any:
    """Auto-detect available vision backend.  Priority: DASHSCOPE > OPENAI > ANTHROPIC."""
    global _vision_client
    if _vision_client is not None:
        return _vision_client

    import os
    if os.environ.get("DASHSCOPE_API_KEY"):
        _vision_client = QwenVisionClient()
    elif os.environ.get("OPENAI_API_KEY"):
        _vision_client = OpenAIVisionClient()
    elif os.environ.get("ANTHROPIC_API_KEY"):
        _vision_client = AnthropicVisionClient()
    else:
        raise RuntimeError(
            "No vision API key found. Set DASHSCOPE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
        )
    return _vision_client


def analyze_scene(
    store: ArtifactStore,
    raster_artifact_id: str,
    question: str,
    *,
    red_band: int = 3,
    green_band: int = 2,
    blue_band: int = 1,
) -> dict[str, Any]:
    """Analyse a geospatial scene using a VLM.

    Renders an RGB preview from the raster artifact, sends it to the VLM
    with *question*, and returns the VLM's textual description.

    Parameters
    ----------
    raster_artifact_id : str
        Artifact ID of the raster to analyse.
    question : str
        Question to ask about the scene (e.g. "Are there any destroyed buildings?").
    """
    from geoharness.schemas import Diagnostic, GeoSkillResult

    source = store.get(raster_artifact_id)
    preview_path = render_rgb_preview(
        source.path,
        store.artifacts_dir / f"{raster_artifact_id}_analysis_rgb.png",
        red_band=red_band,
        green_band=green_band,
        blue_band=blue_band,
    )

    try:
        vlm = _get_vision_client()
        description = vlm.analyze(preview_path, question)
    except Exception as exc:
        return GeoSkillResult(
            status="failed",
            diagnostics=[
                Diagnostic(
                    code="vlm_error",
                    severity="fatal",
                    message=f"VLM analysis failed: {exc}",
                    artifact_id=raster_artifact_id,
                    check_name="analyze_scene",
                )
            ],
        ).to_dict()

    # Register the preview image as an artifact
    preview_artifact = GeoArtifact(
        id=f"{raster_artifact_id}_rgb_preview",
        type="raster",
        path=str(preview_path),
        parents=[raster_artifact_id],
        provenance={"tool": "AnalyzeScene", "question": question},
    )
    store.add(preview_artifact)

    result = GeoSkillResult(
        status="success",
        artifacts=[preview_artifact],
        provenance={
            "tool": "AnalyzeScene",
            "question": question,
            "summary": {"vlm_description": description},
        },
    )
    return result.to_dict()
