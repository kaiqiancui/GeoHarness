# GeoHarness

GeoHarness is a research prototype for **artifact-centric execution and diagnostic feedback** in remote-sensing workflows.  
It models every intermediate product (GeoTIFF, GeoJSON, CSV, report) as a **GeoArtifact** with explicit metadata,
provenance, and structured diagnostics — so that workflow failures are not silent, and deliverables are auditable.

## Core Idea

```text
Ordinary remote-sensing script:
  input files → output result

GeoHarness:
  input artifacts → GeoSkill execution → intermediate artifacts
  → metadata / provenance → diagnostics → recovery / audit
  → deliverable artifacts
```

## Implemented Workflow Families

| Family | Entry Point | Pipeline |
|--------|------------|----------|
| **Measure** | `geoharness/tasks/measure.py` | load → clip → index → stats → report |
| **Detect** | `geoharness/tasks/detect.py` | load → clip → index → mask → vectorize → stats → report |
| **Compare** | `geoharness/tasks/compare.py` | before+after → validate pair → clip both → index both → delta → stats → report |

## Diagnostics Taxonomy

GeoHarness emits structured diagnostics (code, severity, artifact_id, check_name, suggested_actions) covering:

- **Runtime feedback** — file_not_found, invalid_geojson, empty_vector, unsupported_index
- **Geospatial validity** — missing_crs, unsafe_geographic_crs, aoi_outside_raster, crs_mismatch, resolution_mismatch, shape_mismatch
- **Data suitability** — low_valid_pixel_ratio, empty_index, all_nodata
- **Model risk** — empty_mask, saturated_mask, low_positive_mask_ratio
- **Deliverable audit** — missing_provenance, broken_parent_reference, report_value_not_traceable

## Quick Start

```bash
# Install (Python 3.10+)
pip install -e ".[dev]"

# Run tests
python -m pytest

# Measure MVP
python examples/run_measure_mvp.py --workdir runs/measure_mvp

# Artifact graph + Audit
python examples/visualize_artifact_graph.py --metadata runs/measure_mvp/store/metadata.json --output runs/measure_mvp
python examples/audit_deliverables.py --metadata runs/measure_mvp/store/metadata.json --output-dir runs/measure_mvp
```

## Experiments

All experiments write outputs under `runs/`.

| Script | What it does |
|--------|-------------|
| `run_measure_mvp.py` | Measure workflow on synthetic data |
| `run_measure_indices.py` | NDVI + NDWI via same GeoSkill tools |
| `run_detect_mvp.py` | Vegetation + water detection with mask/vectorize |
| `run_compare_mvp.py` | Before/after change analysis (5 cases) |
| `run_failure_injection_suite.py` | 5 controlled failure injections |
| `run_diagnostic_taxonomy_suite.py` | 12-case diagnostic taxonomy stress test |
| `run_ablation.py` | raw_tools vs geoskill_basic vs validators vs feedback_agent |
| `run_feedback_recovery.py` | Diagnostics-driven recovery demo (3 settings × 4 cases) |
| `visualize_artifact_graph.py` | Generate PNG + Mermaid artifact DAG from metadata.json |
| `audit_deliverables.py` | Deliverable file/read/metadata/provenance/report audit |
| `summarize_experiments.py` | Generate experiment_overview + severity_calibration tables |
| `run_oscd_threshold_eval.py` | OSCD delta NDVI threshold baseline + oracle evaluation |
| `run_oscd_cva_eval.py` | OSCD CVA (Change Vector Analysis) percentile baseline |

## OSCD Real-Data Experiment

Download OSCD images and test labels into `data/oscd/raw/` (see `syy/GeoHarness_Project_Completion_Plan.md`
Section 19 for download links), then:

```bash
python examples/prepare_oscd.py
python examples/run_oscd_all.py                    # all 10 test-label cities
python examples/summarize_oscd_results.py           # oscd_summary.csv
python examples/run_oscd_threshold_eval.py          # delta NDVI threshold baseline
python examples/run_oscd_cva_eval.py                # CVA baseline (multi-band)
python examples/visualize_oscd.py --city brasilia
```

## Current Results Summary

- **Measure MVP**: success, 6 artifacts, audit 100% pass
- **Failure Injection**: 5/5 detected
- **Diagnostic Taxonomy**: 12/12 detected (4 taxonomy layers, 100%)
- **Ablation**: raw_tools misses silent failures; validators expose all; feedback_agent recovers AOI outside
- **Feedback Recovery**: AOI outside recovered (warning), missing_band correctly refused
- **Detect MVP**: vegetation + water detection with mask → vectorize → stats → report
- **Compare MVP**: 10 artifacts, audit 100%; all failure cases detected
- **OSCD 10 cities**: 7/10 changed areas show higher abs delta NDVI
- **OSCD Threshold Eval**: delta NDVI mean best F1=0.157, IoU=0.089 (9 fixed thresholds × 10 cities)
- **OSCD CVA Baseline**: Multi-band spectral change, percentile-thresholded, mean best F1=0.292, IoU=0.185 (~1.9× delta NDVI)

## Limitations

GeoHarness is a **lightweight research prototype**, not a production remote-sensing platform:

- It does **not** claim semantic correctness without ground truth.
- It does **not** propose a new remote-sensing model.
- Current OSCD change analysis uses a frozen heuristic (delta NDVI thresholding).
- LLM Agent integration is future work.
- Cloud/season/sensor-suitability diagnostics depend on richer metadata not yet available.

## Repository Structure

```text
geoharness/
  schemas.py            GeoArtifact, Diagnostic, GeoSkillResult
  store.py              ArtifactStore (local filesystem)
  feedback.py           Raster/vector/index validators → diagnostics
  registry.py           Tool registry
  runtime.py            GeoSkill execution runtime
  synthetic.py          Synthetic GeoTIFF + AOI generation
  agents/               FeedbackAgent (rule-based), ScriptedAgent (baseline)
  eval/                 audit.py, metrics.py
  tools/                raster, vector, stats, indices, masks, compare, evaluation, vectorize
  tasks/                measure, detect, compare
  datasets/             oscd.py (OSCD adapter)
  experiments/          fixtures, raw_workflows
examples/               Runnable experiment scripts
tests/                  12 unit tests
runs/                   Experiment outputs
```
