from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OSCD experiment result JSON files.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="runs/oscd_summary.csv")
    parser.add_argument("--markdown-output", default="runs/oscd_summary.md")
    args = parser.parse_args()

    rows = []
    for result_path in sorted(Path(args.runs_dir).glob("oscd_*/oscd_result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload["compare_metrics"]
        rows.append(
            {
                "city": payload["city"],
                "measure_status": payload["measure_status"],
                "changed_pixels": metrics["changed_pixels"],
                "unchanged_pixels": metrics["unchanged_pixels"],
                "mean_delta_ndvi_changed": metrics["mean_delta_ndvi_changed"],
                "mean_delta_ndvi_unchanged": metrics["mean_delta_ndvi_unchanged"],
                "abs_delta_ndvi_changed": metrics["absolute_delta_ndvi_changed"],
                "abs_delta_ndvi_unchanged": metrics["absolute_delta_ndvi_unchanged"],
            }
        )
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    markdown = frame.to_markdown(index=False)
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nwrote {output}")
    print(f"wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
