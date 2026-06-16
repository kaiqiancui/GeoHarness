"""GeoHarness tool definitions in LLM function-calling format.

Each tool is described with a JSON Schema for its parameters (excluding the
``store`` argument which is injected by the runtime).  The ``handler`` field
points to the actual Python function.
"""

from __future__ import annotations

from typing import Any, Callable

from geoharness.llm.vision import analyze_scene
from geoharness.store import ArtifactStore
from geoharness.tools.catalog import list_scenes
from geoharness.tools.compare import change_statistics, compute_delta, validate_raster_pair
from geoharness.tools.crop import crop_view
from geoharness.tools.cva import compute_cva_score
from geoharness.tools.evaluation import evaluate_change_mask, threshold_change_mask
from geoharness.tools.masks import compute_mask_relationship, mask_area_statistics, threshold_mask
from geoharness.tools.raster import clip_by_aoi_artifact, compute_index, load_raster
from geoharness.tools.stats import write_measure_report, zonal_statistics
from geoharness.tools.vector import load_vector
from geoharness.tools.vectorize import vectorize_mask

GeoSkillHandler = Callable[..., Any]

# ── Master tool list ───────────────────────────────────────────────────────────

GEOHARNESS_TOOLS: list[dict[str, Any]] = [
    # ── Catalog tools ────────────────────────────────────────────────────
    {
        "name": "list_scenes",
        "description": (
            "Query the Sentinel-2 satellite catalog for available scenes over an area. "
            "Searches by bounding box and date range. Returns a list of scenes with "
            "date, cloud cover percentage, and scene ID. "
            "Use this to discover available imagery before loading specific scenes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Bounding box [left, bottom, right, top] in WGS84 degrees.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format.",
                },
                "max_cloud": {
                    "type": "integer",
                    "description": "Maximum cloud cover percentage (default 15).",
                },
            },
            "required": ["bbox", "start_date", "end_date"],
            "additionalProperties": False,
        },
        "handler": list_scenes,
    },
    # ── Raster tools ──────────────────────────────────────────────────────
    {
        "name": "load_raster",
        "description": (
            "Load a GeoTIFF raster file and register it as a GeoArtifact. "
            "Returns metadata: CRS, spatial bounds, resolution, array shape, "
            "band names, nodata value, and valid-pixel ratio for band 1. "
            "Use this first to bring a raster into the artifact store."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Unique ID for this artifact (e.g. 'raw_scene').",
                },
                "path": {
                    "type": "string",
                    "description": "Filesystem path to the GeoTIFF file.",
                },
            },
            "required": ["artifact_id", "path"],
            "additionalProperties": False,
        },
        "handler": load_raster,
    },
    {
        "name": "clip_by_aoi",
        "description": (
            "Clip a raster by an AOI vector artifact already in the store. "
            "Returns the clipped raster artifact with updated bounds and shape. "
            "Requires both raster and AOI to be loaded first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the raster to clip.",
                },
                "aoi_id": {
                    "type": "string",
                    "description": "Artifact ID of the AOI vector.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the clipped output raster (e.g. 'clipped_scene').",
                },
            },
            "required": ["raster_id", "aoi_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": clip_by_aoi_artifact,
    },
    {
        "name": "compute_index",
        "description": (
            "Compute a spectral index from a multi-band raster. "
            "Supported indices: NDVI (nir, red), NDWI (green, nir), "
            "NDBI (swir, nir), NBR (nir, swir). "
            "The source raster must have the required bands."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the multi-band source raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the output index raster (e.g. 'ndvi_raster').",
                },
                "index_name": {
                    "type": "string",
                    "description": "Index name: NDVI, NDWI, NDBI, or NBR.",
                    "enum": ["NDVI", "NDWI", "NDBI", "NBR"],
                },
            },
            "required": ["raster_id", "output_id", "index_name"],
            "additionalProperties": False,
        },
        "handler": compute_index,
    },
    # ── Vector tools ──────────────────────────────────────────────────────
    {
        "name": "load_vector",
        "description": (
            "Load a GeoJSON vector file (AOI) and register it as a GeoArtifact. "
            "Returns metadata: geometry count, spatial bounds. "
            "Use this to load an AOI before clipping or zonal statistics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Unique ID for this vector artifact (e.g. 'aoi_vector').",
                },
                "path": {
                    "type": "string",
                    "description": "Filesystem path to the GeoJSON file.",
                },
            },
            "required": ["artifact_id", "path"],
            "additionalProperties": False,
        },
        "handler": load_vector,
    },
    # ── Mask tools ────────────────────────────────────────────────────────
    {
        "name": "threshold_mask",
        "description": (
            "Binarize an index raster into a 0/1 mask by thresholding. "
            "mode='greater' means pixels ABOVE threshold become 1. "
            "mode='less' means pixels BELOW threshold become 1. "
            "Returns mask quality diagnostics (positive pixel ratio, "
            "empty_mask warning if no positives)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index_raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the index raster to threshold.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the output mask raster.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Index value threshold (default 0.3).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["greater", "less"],
                    "description": "Keep pixels 'greater' or 'less' than threshold.",
                },
                "mask_name": {
                    "type": "string",
                    "description": "Label for the mask (e.g. 'vegetation', 'water').",
                },
            },
            "required": ["index_raster_id", "output_id", "threshold", "mode", "mask_name"],
            "additionalProperties": False,
        },
        "handler": threshold_mask,
    },
    {
        "name": "mask_area_statistics",
        "description": (
            "Compute area statistics for a binary mask. "
            "Returns valid pixel count, positive pixel count, pixel ratio, "
            "and estimated area (in m² if projected CRS)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mask_raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the binary mask raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the output statistics table.",
                },
            },
            "required": ["mask_raster_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": mask_area_statistics,
    },
    {
        "name": "compute_mask_relationship",
        "description": (
            "Compute spatial relationship metrics between two binary masks. "
            "Calculates intersection pixels, union pixels, Intersection-over-Union (IoU), "
            "coverage_a (fraction of mask A pixels overlapped by mask B), "
            "and coverage_b (fraction of mask B pixels overlapped by mask A). "
            "Use this to compare pre- and post-disaster building masks to quantify damage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mask_a_id": {
                    "type": "string",
                    "description": "Artifact ID of the first binary mask (e.g. pre-disaster building mask).",
                },
                "mask_b_id": {
                    "type": "string",
                    "description": "Artifact ID of the second binary mask (e.g. post-disaster building mask).",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the output metrics table.",
                },
            },
            "required": ["mask_a_id", "mask_b_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": compute_mask_relationship,
    },
    {
        "name": "vectorize_mask",
        "description": (
            "Convert a binary mask raster to polygon vector features. "
            "Useful for extracting region boundaries from a detection mask."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mask_raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the binary mask raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the output vector (e.g. 'vegetation_regions').",
                },
            },
            "required": ["mask_raster_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": vectorize_mask,
    },
    {
        "name": "crop_view",
        "description": (
            "Crop a rectangular sub-view from a raster for spatial navigation. "
            "Use this to zoom into a specific region, shift the viewport, or "
            "check whether targets at the edge of the current view extend "
            "beyond the boundary. Returns which edges the crop touches "
            "(top_edge, left_edge, bottom_edge, right_edge)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the source raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the cropped output.",
                },
                "x": {
                    "type": "integer",
                    "description": "Left pixel coordinate (default 0).",
                },
                "y": {
                    "type": "integer",
                    "description": "Top pixel coordinate (default 0).",
                },
                "width": {
                    "type": "integer",
                    "description": "Crop width in pixels (default 320).",
                },
                "height": {
                    "type": "integer",
                    "description": "Crop height in pixels (default 320).",
                },
            },
            "required": ["raster_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": crop_view,
    },
    # ── Statistics tools ──────────────────────────────────────────────────
    {
        "name": "zonal_statistics",
        "description": (
            "Compute zonal statistics (mean, min, max, std) for a raster. "
            "If the raster was clipped by an AOI, statistics are for that zone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the raster to summarize.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the statistics table.",
                },
            },
            "required": ["raster_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": zonal_statistics,
    },
    {
        "name": "write_report",
        "description": (
            "Generate a human-readable Markdown report from a statistics table. "
            "This is typically the final step of a workflow."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_id": {
                    "type": "string",
                    "description": "Artifact ID of the statistics table (from zonal_statistics or mask_area_statistics).",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the report.",
                },
                "title": {
                    "type": "string",
                    "description": "Report title.",
                },
            },
            "required": ["table_id", "output_id", "title"],
            "additionalProperties": False,
        },
        "handler": write_measure_report,
    },
    # ── Compare / change-detection tools ─────────────────────────────────
    {
        "name": "validate_raster_pair",
        "description": (
            "Check whether two rasters are compatible for comparison. "
            "Verifies CRS match, resolution compatibility, shape match, "
            "bounds overlap, and band availability. "
            "Call this before computing delta between before/after rasters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "before_id": {
                    "type": "string",
                    "description": "Artifact ID of the 'before' raster.",
                },
                "after_id": {
                    "type": "string",
                    "description": "Artifact ID of the 'after' raster.",
                },
            },
            "required": ["before_id", "after_id"],
            "additionalProperties": False,
        },
        "handler": validate_raster_pair,
    },
    {
        "name": "compute_delta",
        "description": (
            "Compute pixel-wise delta (after - before) between two index rasters. "
            "Both rasters must have the same shape. "
            "Returns a delta raster with change magnitude per pixel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "before_index_id": {
                    "type": "string",
                    "description": "Artifact ID of the before index raster.",
                },
                "after_index_id": {
                    "type": "string",
                    "description": "Artifact ID of the after index raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the delta raster.",
                },
                "metric_name": {
                    "type": "string",
                    "description": "Display name (e.g. 'delta_ndvi').",
                },
            },
            "required": ["before_index_id", "after_index_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": compute_delta,
    },
    {
        "name": "change_statistics",
        "description": (
            "Compute descriptive statistics for a delta raster. "
            "Reports mean/median/min/max/std delta, positive/negative pixel counts, "
            "and count of pixels exceeding a large-change threshold."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delta_id": {
                    "type": "string",
                    "description": "Artifact ID of the delta raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the statistics table.",
                },
                "large_change_threshold": {
                    "type": "number",
                    "description": "Absolute delta above this is 'large change' (default 0.2).",
                },
            },
            "required": ["delta_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": change_statistics,
    },
    {
        "name": "compute_cva_score",
        "description": (
            "Compute Change Vector Analysis (CVA) score: per-pixel Euclidean "
            "distance across ALL bands between before and after rasters. "
            "This produces a continuous change-intensity raster. "
            "More sensitive than single-index delta because it uses full spectral information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "before_raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the before multi-band raster.",
                },
                "after_raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the after multi-band raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the CVA score raster.",
                },
            },
            "required": ["before_raster_id", "after_raster_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": compute_cva_score,
    },
    {
        "name": "threshold_change_mask",
        "description": (
            "Create a binary change mask from a delta or CVA score raster "
            "by thresholding the absolute value.  Used to separate changed "
            "from unchanged pixels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delta_raster_id": {
                    "type": "string",
                    "description": "Artifact ID of the delta or CVA score raster.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the binary change mask.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Pixels with abs(value) > threshold are marked as changed.",
                },
            },
            "required": ["delta_raster_id", "output_id", "threshold"],
            "additionalProperties": False,
        },
        "handler": threshold_change_mask,
    },
    {
        "name": "evaluate_change_mask",
        "description": (
            "Evaluate a predicted change mask against an oracle (ground-truth) label. "
            "Computes precision, recall, F1, IoU, and accuracy. "
            "The oracle label is stored as a hidden artifact (oracle_only=True) "
            "and is NOT visible to the agent during planning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_id": {
                    "type": "string",
                    "description": "Artifact ID of the predicted change mask.",
                },
                "oracle_id": {
                    "type": "string",
                    "description": "Artifact ID of the oracle label.",
                },
                "output_id": {
                    "type": "string",
                    "description": "Artifact ID for the metrics table.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold value used (for record-keeping).",
                },
            },
            "required": ["predicted_id", "oracle_id", "output_id"],
            "additionalProperties": False,
        },
        "handler": evaluate_change_mask,
    },
    {
        "name": "analyze_scene",
        "description": (
            "Analyse a geospatial scene image using AI vision. "
            "Renders an RGB preview from a raster artifact and asks a VLM "
            "(GPT-4o / Claude Vision) a question about what it sees. "
            "Use this for qualitative scene understanding: detecting objects, "
            "assessing damage, describing land cover, checking for buildings, etc. "
            "Returns a text description from the vision model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raster_artifact_id": {
                    "type": "string",
                    "description": "Artifact ID of the raster to visually analyse.",
                },
                "question": {
                    "type": "string",
                    "description": (
                        "What to ask about the scene. Be specific. Examples: "
                        "'Are there any football fields in this image? List their bounding boxes.', "
                        "'Describe the extent of building damage visible in this post-disaster scene.', "
                        "'Is there any target touching the edge of the image?'"
                    ),
                },
            },
            "required": ["raster_artifact_id", "question"],
            "additionalProperties": False,
        },
        "handler": analyze_scene,
    },
]


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Look up a tool definition by name."""
    for tool in GEOHARNESS_TOOLS:
        if tool["name"] == name:
            return tool
    return None


def get_handler(name: str) -> GeoSkillHandler | None:
    """Return the Python handler function for *name*."""
    tool = get_tool_by_name(name)
    if tool is None:
        return None
    return tool["handler"]


def llm_tool_schemas() -> list[dict[str, Any]]:
    """Return tool definitions stripped of the internal ``handler`` field."""
    return [
        {k: v for k, v in tool.items() if k != "handler"}
        for tool in GEOHARNESS_TOOLS
    ]
