"""Exp 1: Harness Ablation — raw tools vs harness without diagnostics vs full harness.

3 settings × 5 tasks × 3 repetitions = 45 agent runs.

Settings:
  A: Raw tools (bare rasterio/numpy, no harness)
  B: GeoHarness, diagnostics hidden from agent
  C: GeoHarness full (tools + diagnostics visible)

Tasks:
  1. Normal NDVI analysis (no error injected)
  2. Missing CRS
  3. AOI outside raster
  4. Missing NIR band
  5. High nodata (75% invalid)

Metrics:
  - Task success rate
  - Silent failure rate (agent claims success but answer is wrong)
  - Error detection rate (agent mentions the anomaly)
  - Recovery rate (agent attempts corrective action)

Usage::

    python examples/run_exp1_ablation.py
    python examples/run_exp1_ablation.py --repetitions 1  # quick test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.agent import GeoHarnessAgent
from geoharness.experiments.fixtures import (
    copy_with_high_nodata,
    copy_without_band,
    copy_without_crs,
    write_aoi_geojson,
)
from geoharness.experiments.raw_tools import RAW_HANDLERS, RAW_TOOL_DEFS
from geoharness.llm.client import create_client
from geoharness.llm.tools import llm_tool_schemas
from geoharness.synthetic import write_synthetic_measure_fixture

# ── Task definitions ───────────────────────────────────────────────────────────

TASKS = {
    "normal": {
        "label": "Normal NDVI analysis",
        "has_error": False,
        "expected_error_codes": [],
        "expected_answer_markers": ["mean", "ndvi", "0."],
    },
    "missing_crs": {
        "label": "Missing CRS",
        "has_error": True,
        "expected_error_codes": ["missing_crs"],
        "expected_answer_markers": ["missing", "crs", "coordinate", "projection", "reference"],
    },
    "aoi_outside": {
        "label": "AOI outside raster",
        "has_error": True,
        "expected_error_codes": ["aoi_outside_raster"],
        "expected_answer_markers": ["outside", "bounds", "aoi", "overlap", "not within", "beyond"],
    },
    "missing_band": {
        "label": "Missing NIR band",
        "has_error": True,
        "expected_error_codes": ["missing_band"],
        "expected_answer_markers": ["missing", "band", "nir", "not available", "cannot"],
    },
    "high_nodata": {
        "label": "High nodata ratio",
        "has_error": True,
        "expected_error_codes": ["low_valid_pixel_ratio"],
        "expected_answer_markers": ["nodata", "invalid", "quality", "low", "ratio", "warning", "valid"],
    },
}


# ── Metrics scoring ────────────────────────────────────────────────────────────


def score_result(result: dict, task_def: dict) -> dict[str, bool | float]:
    """Compute per-run metrics from agent result + task definition."""
    answer = (result.get("answer") or "").lower()
    trace = result.get("trace") or []
    steps = result.get("steps", 0)

    # Did the agent finish without hitting max_steps?
    completed = result["status"] == "success"

    # Did the agent mention the expected anomaly?
    error_mentioned = False
    if task_def["has_error"]:
        markers = task_def.get("expected_answer_markers", [])
        error_mentioned = any(m in answer for m in markers)

    # Did diagnostics fire the expected code?
    expected_codes = task_def.get("expected_error_codes", [])
    observed_codes: list[str] = []
    for entry in trace:
        observed_codes.extend(entry.get("diagnostic_codes", []))
    diagnostic_hit = any(c in observed_codes for c in expected_codes) if expected_codes else True

    # Was there a recovery attempt?
    recovery = any("retry" in str(entry).lower() or "recover" in str(entry).lower() for entry in trace)

    # Task success: completed + (no error expected OR error was detected)
    if task_def["has_error"]:
        task_ok = completed and (error_mentioned or diagnostic_hit)
    else:
        task_ok = completed

    # Silent failure: completed but didn't detect the error
    silent_failure = completed and task_def["has_error"] and not error_mentioned and not diagnostic_hit

    return {
        "completed": completed,
        "task_success": task_ok,
        "silent_failure": silent_failure,
        "error_mentioned": error_mentioned,
        "diagnostic_hit": diagnostic_hit,
        "recovery_attempted": recovery,
        "steps": steps,
        "observed_codes": ";".join(observed_codes),
    }


# ── Raw tool handler wrapper ──────────────────────────────────────────────────

def _wrap_raw_handler(fn):
    """Wrap a raw tool handler so it accepts (store, **kwargs) and ignores store."""
    def wrapped(store, **kwargs):
        return fn(**kwargs)
    return wrapped


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 1: Harness ablation experiment.")
    parser.add_argument("--workdir", default="runs/exp1_ablation")
    parser.add_argument("--backend", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--skip-settings", default="", help="Comma-separated: raw,nodiag,full")
    parser.add_argument("--skip-tasks", default="", help="Comma-separated task keys")
    args = parser.parse_args()

    skip_settings = set(args.skip_settings.split(",")) if args.skip_settings else set()
    skip_tasks = set(args.skip_tasks.split(",")) if args.skip_tasks else set()

    workdir = Path(args.workdir)
    inputs_dir = workdir / "inputs"
    base_raster, base_aoi = write_synthetic_measure_fixture(inputs_dir / "base")

    # Prepare error-injected variants
    no_crs_raster = copy_without_crs(base_raster, inputs_dir / "missing_crs" / "scene.tif")
    missing_band_raster = copy_without_band(base_raster, inputs_dir / "missing_band" / "scene.tif")
    high_nodata_raster = copy_with_high_nodata(base_raster, inputs_dir / "high_nodata" / "scene.tif")
    outside_aoi = write_aoi_geojson((900000, 900, 900100, 1000), inputs_dir / "outside_aoi.geojson")

    task_inputs = {
        "normal": (base_raster, base_aoi),
        "missing_crs": (no_crs_raster, base_aoi),
        "aoi_outside": (base_raster, outside_aoi),
        "missing_band": (missing_band_raster, base_aoi),
        "high_nodata": (high_nodata_raster, base_aoi),
    }

    # Raw tool definitions (strip internal name for LLM schema)
    raw_schemas = [{k: v for k, v in t.items()} for t in RAW_TOOL_DEFS]
    raw_handlers = {name: _wrap_raw_handler(fn) for name, fn in RAW_HANDLERS.items()}

    client = create_client(args.backend, model=args.model)
    all_rows: list[dict] = []

    # Shorthand helper
    def _task_text(task_key: str) -> str:
        raster, aoi = task_inputs[task_key]
        return (
            f"Your task is to analyse the vegetation condition in the provided raster "
            f"using NDVI. The raster file is at: {raster}\n"
            f"The AOI file is at: {aoi}\n\n"
            f"Steps: load the raster, load the AOI, clip, compute NDVI, compute statistics. "
            f"If you encounter any problems, describe them clearly in your answer. "
            f"In your final answer, report the mean NDVI value and note any data quality issues."
        )

    for task_key, task_def in TASKS.items():
        if task_key in skip_tasks:
            continue

        for rep in range(1, args.repetitions + 1):
            print(f"\n{'=' * 60}")
            print(f"Task: {task_def['label']}  |  Repetition {rep}/{args.repetitions}")
            print(f"{'=' * 60}")

            # ── Setting A: Raw tools ────────────────────────────────────
            if "raw" not in skip_settings:
                print(f"\n  [Setting A: Raw tools]")
                agent = GeoHarnessAgent(
                    client,
                    store_root=workdir / "raw" / task_key / f"rep_{rep}",
                    max_steps=15,
                    verbose=False,
                    diagnostics_visible=True,
                    custom_tools=raw_schemas,
                    custom_handlers=raw_handlers,
                )
                result = agent.run(_task_text(task_key))
                scores = score_result(result, task_def)
                all_rows.append({
                    "setting": "A_raw", "task": task_key, "rep": rep,
                    **scores, "answer_preview": (result.get("answer") or "")[:200],
                })

            # ── Setting B: Harness, diagnostics hidden ──────────────────
            if "nodiag" not in skip_settings:
                print(f"\n  [Setting B: Harness, no diagnostics]")
                agent = GeoHarnessAgent(
                    client,
                    store_root=workdir / "nodiag" / task_key / f"rep_{rep}",
                    max_steps=15,
                    verbose=False,
                    diagnostics_visible=False,
                )
                result = agent.run(_task_text(task_key))
                scores = score_result(result, task_def)
                all_rows.append({
                    "setting": "B_nodiag", "task": task_key, "rep": rep,
                    **scores, "answer_preview": (result.get("answer") or "")[:200],
                })

            # ── Setting C: Full harness ─────────────────────────────────
            if "full" not in skip_settings:
                print(f"\n  [Setting C: Full harness]")
                agent = GeoHarnessAgent(
                    client,
                    store_root=workdir / "full" / task_key / f"rep_{rep}",
                    max_steps=15,
                    verbose=False,
                    diagnostics_visible=True,
                )
                result = agent.run(_task_text(task_key))
                scores = score_result(result, task_def)
                all_rows.append({
                    "setting": "C_full", "task": task_key, "rep": rep,
                    **scores, "answer_preview": (result.get("answer") or "")[:200],
                })

    # ── Aggregate and export ────────────────────────────────────────────────

    frame = pd.DataFrame(all_rows)

    # Per-setting aggregation
    agg = frame.groupby(["setting", "task"]).agg(
        task_success_rate=("task_success", "mean"),
        silent_failure_rate=("silent_failure", "mean"),
        error_mention_rate=("error_mentioned", "mean"),
        recovery_rate=("recovery_attempted", "mean"),
        avg_steps=("steps", "mean"),
    ).round(3).reset_index()

    # Overall per-setting aggregation
    overall = frame.groupby("setting").agg(
        task_success_rate=("task_success", "mean"),
        silent_failure_rate=("silent_failure", "mean"),
        error_mention_rate=("error_mentioned", "mean"),
        recovery_rate=("recovery_attempted", "mean"),
        avg_steps=("steps", "mean"),
    ).round(3).reset_index()

    # Write outputs
    detail_csv = workdir / "exp1_detail.csv"
    agg_csv = workdir / "exp1_aggregate.csv"
    overall_csv = workdir / "exp1_overall.csv"
    workdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(detail_csv, index=False)
    agg.to_csv(agg_csv, index=False)
    overall.to_csv(overall_csv, index=False)

    print(f"\n{'=' * 60}")
    print("  Exp 1 Results — Aggregate by Setting × Task")
    print(f"{'=' * 60}")
    print(agg.to_markdown(index=False))

    print(f"\n{'=' * 60}")
    print("  Exp 1 Results — Overall by Setting")
    print(f"{'=' * 60}")
    print(overall.to_markdown(index=False))

    print(f"\nDetail: {detail_csv}")
    print(f"Aggregate: {agg_csv}")
    print(f"Overall: {overall_csv}")


if __name__ == "__main__":
    main()
