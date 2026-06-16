"""Case Study 1: Multi-spectral temporal urban built-up area judgment.

Agent uses NDBI (Normalized Difference Built-up Index) to compare urban
characteristics across 3 Sentinel-2 time steps for Brasilia.

Data: Sentinel-2 L2A from Planetary Computer (pre-downloaded by prepare_case1_data.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.agent import GeoHarnessAgent
from geoharness.llm.client import create_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Case Study 1: Temporal urban NDBI analysis.")
    parser.add_argument("--workdir", default="runs/case1_temporal")
    parser.add_argument("--backend", default="deepseek")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    data_dir = Path("data/case1_temporal")

    # Discover available scenes
    ms_files = sorted(data_dir.glob("brasilia_*_ms.tif"))
    if len(ms_files) < 2:
        print("ERROR: Need at least 2 multispectral scenes. Run prepare_case1_data.py first.")
        print(f"Found: {ms_files}")
        sys.exit(1)

    file_list = "\n".join(f"  {f}" for f in ms_files)

    task = f"""You are analysing urban expansion in Brasilia, Brazil using Sentinel-2 satellite imagery.

Available multi-spectral GeoTIFF files (each has 5 bands: blue, green, red, nir, swir):
{file_list}

Your task: Determine which time period shows the most significant built-up/urban characteristics.

Method:
1. Use list_scenes to discover available Sentinel-2 imagery for Brasilia
   (bbox: [-47.95, -15.85, -47.85, -15.75], dates: 2020-01-01 to 2020-12-31)
2. Load each available scene as a raster artifact
3. Compute NDBI (Normalized Difference Built-up Index) for EACH scene.
   NDBI uses swir and nir bands — higher NDBI means more built-up/urban area.
4. Get zonal statistics for each NDBI raster (mean, max, p95)
5. Compare the NDBI values across time periods.
   Higher values = more built-up/urban signature.
6. Conclude which period has the strongest urban characteristics, with evidence.

Important: NDBI values for urban areas are typically positive but small (0.0 to 0.3).
Negative values indicate vegetation or water. The relative comparison across time
periods matters more than the absolute values."""

    print("=" * 60)
    print("Case Study 1: Multi-temporal Urban NDBI Analysis")
    print("=" * 60)
    print(f"Scenes available: {len(ms_files)}")
    for f in ms_files:
        print(f"  {f.name}")

    client = create_client(args.backend, model=args.model)
    agent = GeoHarnessAgent(client, store_root=workdir / "store", max_steps=20, verbose=True)
    result = agent.run(task)

    # Save results
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "agent_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # Write summary
    summary = f"""# Case Study 1: Multi-temporal Urban NDBI Analysis

## Experiment Design
- **Data**: Sentinel-2 L2A, {len(ms_files)} scenes over Brasilia (2020)
- **Index**: NDBI (Normalized Difference Built-up Index) = (SWIR - NIR) / (SWIR + NIR)
- **Bands**: B02, B03, B04, B08 (10m) + B11 SWIR (20m, resampled to 10m)
- **Agent**: DeepSeek-chat with GeoHarness full harness

## Available Scenes
"""
    for f in ms_files:
        summary += f"- {f.name}\n"

    summary += f"""
## Agent Execution
- Status: {result['status']}
- Steps: {result['steps']}
- Metadata: {result['metadata_path']}

## Agent Answer
{result['answer']}

## Execution Trace
"""
    for entry in result.get("trace", []):
        summary += f"- Step {entry['step']}: {entry['tool_name']} → {entry['status']}"
        if entry.get("diagnostic_codes"):
            summary += f" [{', '.join(entry['diagnostic_codes'])}]"
        summary += "\n"

    (workdir / "RESULTS.md").write_text(summary, encoding="utf-8")
    print(f"\nResults saved to {workdir}/")


if __name__ == "__main__":
    main()
