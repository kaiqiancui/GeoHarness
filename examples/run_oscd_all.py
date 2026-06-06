from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.datasets.oscd import extract_zip_once, find_oscd_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSCD experiments for all cities with test labels.")
    parser.add_argument("--raw-dir", default="data/oscd/raw")
    parser.add_argument("--extract-dir", default="data/oscd/extracted")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    for zip_path in sorted(raw_dir.glob("*.zip")):
        extract_zip_once(zip_path, extract_dir)
    labels_root = find_oscd_root(extract_dir, "Labels")
    cities = sorted(path.name for path in labels_root.iterdir() if path.is_dir())
    if not cities:
        raise RuntimeError(f"No OSCD label cities found under {labels_root}")

    script = Path(__file__).resolve().parent / "run_oscd_experiment.py"
    for city in cities:
        workdir = Path(args.runs_dir) / f"oscd_{city}"
        command = [
            sys.executable,
            str(script),
            "--city",
            city,
            "--raw-dir",
            str(raw_dir),
            "--extract-dir",
            str(extract_dir),
            "--workdir",
            str(workdir),
        ]
        print(f"running {city} -> {workdir}")
        subprocess.run(command, check=True)

    summarize_script = Path(__file__).resolve().parent / "summarize_oscd_results.py"
    subprocess.run(
        [
            sys.executable,
            str(summarize_script),
            "--runs-dir",
            args.runs_dir,
            "--output",
            str(Path(args.runs_dir) / "oscd_summary.csv"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
