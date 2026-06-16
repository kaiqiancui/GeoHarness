# GeoHarness

An execution harness for LLM-driven remote sensing agents.  GeoHarness wraps
geospatial tools in a structured interface with artifact provenance tracking
and diagnostic feedback, enabling LLM agents to execute multi-step earth
observation workflows reliably.

## Core Idea

```
Bare LLM + Python scripts:
  prompt → code → output  (silent failures possible)

GeoHarness:
  prompt → agent loop → GeoSkill tools → GeoArtifacts
         → provenance chain → diagnostic feedback → auditable result
```

The key insight: the same LLM achieves 80% task success with raw rasterio
functions, but 100% with GeoHarness normalized tools and diagnostic feedback.
The gap comes from tool interface quality, not model size.

## Architecture

```
LLM Agent (DeepSeek / OpenAI / Anthropic)
  → ReAct loop: observe → plan → call tool → interpret feedback → iterate
     ↓
GeoHarness Execution Layer
  ├─ 17 GeoSkill tools (load, clip, index, mask, compare, report, ...)
  ├─ Artifact provenance (every intermediate result tracked)
  └─ Diagnostic engine (12+ standardized codes across 4 taxonomy layers)
     ↓
GIS Backend (rasterio / numpy)
```

## Quick Start

```bash
pip install -e ".[dev]"

# Set API keys in geoharness/.env:
#   DEEPSEEK_API_KEY=sk-...
#   DASHSCOPE_API_KEY=sk-...  (for VLM analysis)

# Run an agent-driven NDVI analysis on synthetic data
python examples/run_agent_measure.py
```

## Experiments

| Experiment | Script | What it tests |
|-----------|--------|--------------|
| **Exp 1: Ablation** | `run_exp1_ablation.py` | Fixed script vs raw tools + Agent vs Harness + Agent (4 settings × 5 tasks) |
| **Case 1: Temporal NDBI** | `run_case1_temporal.py` | Agent autonomously selects NDBI over NDVI, compares 3 Sentinel-2 scenes |
| **Case 2: Spatial Navigation** | `run_case2_football.py` | Agent uses crop_view + VLM to systematically explore satellite imagery |
| **Case 3: Disaster Assessment** | `run_case3_turkey.py` | Agent fuses quantitative (IoU) and qualitative (VLM) evidence for damage rating |

Data preparation scripts: `prepare_case1_data.py` (Planetary Computer Sentinel-2), `download_case2_data.py` (Google Maps).

Key finding from Exp 1: the same DeepSeek model achieves 40% (fixed script) → 80% (raw tools) → 100% (Harness full) success rate, with 44% fewer steps.

## Repository Structure

```
geoharness/
  agent.py              GeoHarnessAgent — ReAct agent loop
  schemas.py            GeoArtifact, Diagnostic, GeoSkillResult
  store.py              ArtifactStore (JSON metadata + JSONL diagnostics)
  feedback.py           Raster/vector/index validators → diagnostic codes
  registry.py           Tool registry
  runtime.py            GeoSkill execution runtime
  synthetic.py          Synthetic GeoTIFF + AOI generation
  llm/
    client.py           LLM client (DeepSeek / OpenAI / Anthropic)
    tools.py            17 tool definitions in function-calling format
    context.py          Artifact summaries + tool result formatting
    vision.py           VLM client (Qwen3-VL) + analyze_scene tool
  tools/
    raster.py           load_raster, clip_by_aoi, compute_index
    vector.py           load_vector
    masks.py            threshold_mask, mask_area_statistics, compute_mask_relationship
    compare.py          validate_raster_pair, compute_delta, change_statistics
    crop.py             crop_view (spatial navigation)
    catalog.py          list_scenes (Planetary Computer STAC API)
    cva.py              compute_cva_score
    stats.py            zonal_statistics, write_measure_report
    evaluation.py       threshold_change_mask, evaluate_change_mask
    indices.py          NDVI, NDWI, NDBI, NBR specifications
    vectorize.py        vectorize_mask
  experiments/
    fixtures.py         Error injection fixtures
    raw_tools.py        Bare rasterio functions (ablation baseline)
  datasets/
    oscd.py             OSCD dataset adapter
examples/               Runnable experiment scripts
tests/                  12 unit tests
```

## Diagnostics Taxonomy

| Layer | Example codes |
|-------|-------------|
| Runtime | `file_not_found`, `invalid_geojson`, `unsupported_index` |
| Geospatial | `missing_crs`, `aoi_outside_raster`, `crs_mismatch` |
| Data Quality | `low_valid_pixel_ratio`, `empty_index` |
| Model Risk | `empty_mask`, `saturated_mask` |

Each diagnostic includes `code`, `severity`, `message`, and `suggested_actions`
— structured feedback the agent can act on without parsing Python tracebacks.
