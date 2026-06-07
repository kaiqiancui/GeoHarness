from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from geoharness.eval.audit import audit_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit deliverable artifacts from metadata.json.")
    parser.add_argument("--metadata", default="runs/measure_mvp/store/metadata.json", help="Path to metadata.json")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to parent of metadata parent)")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir) if args.output_dir else metadata_path.parent.parent

    result = audit_artifacts(metadata_path)
    rows = result["rows"]
    summary = result["summary"]

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "audit_results.csv"
    md_path = output_dir / "audit_results.md"
    json_path = output_dir / "audit_results.json"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")
    json_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")

    print(frame.to_markdown(index=False))
    print()
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
