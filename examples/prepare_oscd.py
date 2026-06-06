from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.datasets.oscd import extract_zip_once, find_oscd_root, list_cities  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract local OSCD zip files and list cities.")
    parser.add_argument("--raw-dir", default="data/oscd/raw")
    parser.add_argument("--extract-dir", default="data/oscd/extracted")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    for zip_path in sorted(raw_dir.glob("*.zip")):
        extract_zip_once(zip_path, extract_dir)
        print(f"extracted: {zip_path}")

    images_root = find_oscd_root(extract_dir, "Images")
    labels_root = find_oscd_root(extract_dir, "Labels")
    print(f"images_root={images_root}")
    print(f"labels_root={labels_root}")
    print("cities=" + ",".join(list_cities(images_root)))


if __name__ == "__main__":
    main()
