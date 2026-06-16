"""Case Study 3: Building damage assessment — 2023 Turkey earthquake (Antakya).

Uses cleaned Maxar WV3 imagery with identical valid-data masks.
"""

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.agent import GeoHarnessAgent
from geoharness.llm.client import create_client
from geoharness.llm.vision import render_rgb_preview

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="runs/case3_turkey")
    parser.add_argument("--backend", default="deepseek")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    pre_path = "data/turkey_eq/antakya_pre_clean.tif"
    post_path = "data/turkey_eq/antakya_post_clean.tif"

    # Generate RGB previews
    for p in [pre_path, post_path]:
        preview = render_rgb_preview(p, workdir / "inputs" / Path(p).name.replace('.tif', '_rgb.png'))
        print(f"Preview: {preview}")

    task = f"""You are assessing building damage from the 2023 Turkey earthquake (M7.8, Feb 6, 2023)
in Antakya (Hatay province).

Pre-disaster image: {pre_path} (Jan 7, 2023, Maxar WV3, 0.3m resolution)
Post-disaster image: {post_path} (Feb 9, 2023, 3 days after earthquake)

Both images have been preprocessed to mask out invalid pixels (black=0=nodata).
Only pixels with valid data in BOTH images are included.

Steps:
1. Load both images as raster artifacts
2. Use threshold_mask to extract bright urban surfaces from both (threshold ~120)
3. Compare the two masks using compute_mask_relationship (IoU, coverage)
4. Use mask_area_statistics to quantify the area change
5. Use analyze_scene on post-disaster to visually describe damage
6. Synthesize quantitative (IoU, area change, coverage) and qualitative (VLM) evidence
7. Give a final damage severity rating with justification

Important notes:
- Black pixels (value=0) are masked nodata — they will NOT be counted by threshold_mask
- Bright surfaces include buildings (pre) and buildings+debris (post)
- A large increase in bright area + low IoU = debris from collapsed buildings
"""

    print("=" * 60)
    print("Case Study 3: Turkey Earthquake Building Damage Assessment")
    print("=" * 60)

    client = create_client(args.backend)
    agent = GeoHarnessAgent(client, store_root=workdir / "store", max_steps=25, verbose=True)
    result = agent.run(task)

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "agent_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    summary = f"""# Case Study 3: Turkey Earthquake Building Damage

## Data
- Pre: data/turkey_eq/antakya_pre_clean.tif (Maxar WV3, 2023-01-07)
- Post: data/turkey_eq/antakya_post_clean.tif (Maxar WV3, 2023-02-09)
- Both cropped to valid-data intersection, black pixels masked

## Results
- Status: {result['status']}
- Steps: {result['steps']}

## Agent Answer
{result['answer']}

## Trace
"""
    for t in result.get("trace", []):
        codes = t.get("diagnostic_codes", [])
        summary += f"- Step {t['step']}: {t['tool_name']} -> {t['status']}" + (f" [{','.join(codes)}]" if codes else "") + "\n"

    (workdir / "RESULTS.md").write_text(summary)
    print(f"\nResults saved to {workdir}/")

if __name__ == "__main__":
    main()
