"""Scripted agent — fixed Measure plan, no feedback-driven recovery.

This is the no-feedback baseline. The agent executes the Measure workflow
to completion regardless of diagnostic warnings (it only stops on fatal errors,
which is the default behaviour of run_measure_workflow).
"""

from __future__ import annotations

from pathlib import Path

from geoharness.tasks.measure import run_measure_workflow


def run_scripted_agent(
    *,
    store_root: str | Path,
    raster_path: str | Path,
    aoi_path: str | Path,
) -> dict:
    """Execute fixed Measure plan. No recovery on diagnostics."""
    result = run_measure_workflow(
        store_root=store_root,
        raster_path=raster_path,
        aoi_path=aoi_path,
    )
    result["agent"] = "ScriptedAgent"
    result["trace"] = [
        {
            "step": "measure_workflow",
            "decision": "execute_fixed_plan",
            "status": result["status"],
            "diagnostics": [d["code"] for d in result["diagnostics"]],
        }
    ]
    return result
