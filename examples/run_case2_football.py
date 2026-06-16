"""Case Study 2: Moving field-of-view edge target counting.

Agent uses spatial navigation (crop_view) + VLM (analyze_scene) to count
completely visible football fields in high-resolution satellite imagery.

Data: Google Maps satellite views of Qingdao Guoxin Sports Center.
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
    parser = argparse.ArgumentParser(description="Case Study 2: Football field edge counting.")
    parser.add_argument("--workdir", default="runs/case2_football")
    parser.add_argument("--backend", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    data_dir = Path("data/case2_football")
    full_scene = data_dir / "full_scene.tif"
    shifted_view = data_dir / "shifted_view.tif"
    zoomed_out = data_dir / "zoomed_out.tif"

    task = f"""You are exploring high-resolution satellite imagery of a sports complex.

Available images (use exact paths):
  Full scene: {full_scene} (640x640)
  Shifted view: {shifted_view} (640x640, shifted position)

Your task: Systematically explore the area to identify sports fields.

Method:
1. Load the full scene
2. Use crop_view to examine different regions (e.g., split into quadrants)
3. For each cropped view, use analyze_scene to ask the VLM: "Describe what you see. Are there any rectangular green/brown sports fields? Is anything touching the edge?"
4. If a crop touches an edge and might have cut-off content, shift your view using crop_view with different x,y coordinates to see the adjacent area
5. Load the shifted_view and compare - check areas that were at edges of the full scene

After exploring, give your best assessment. This is about demonstrating the SPATIAL NAVIGATION process (systematic exploration, edge checking, view shifting) more than getting a perfect count. Describe what you found."""

    print("=" * 60)
    print("Case Study 2: Football Field Edge Counting")
    print("=" * 60)
    print(f"\nTask: Count completely visible football fields")
    print(f"Full scene: {full_scene}")
    print()

    client = create_client(args.backend, model=args.model)
    agent = GeoHarnessAgent(client, store_root=workdir / "store", max_steps=args.max_steps, verbose=True)
    result = agent.run(task)

    # Save results
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "agent_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # Write summary
    summary = f"""# Case Study 2: Football Field Edge Counting

## Experiment Design
- **Data**: Google Maps satellite imagery of Qingdao Guoxin Sports Center
- **VLM**: Qwen3-VL (via DashScope) for visual football field detection
- **Spatial Navigation**: crop_view tool for viewport shifting/zooming
- **Agent**: DeepSeek-chat with GeoHarness full harness

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
