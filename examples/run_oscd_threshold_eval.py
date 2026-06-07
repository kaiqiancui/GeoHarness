"""OSCD delta NDVI threshold baseline — standard pixel-level oracle evaluation.

For each OSCD test-label city:
  1. Compute before/after NDVI and delta NDVI rasters.
  2. Register delta, oracle label, predicted masks, and evaluation metrics as
     GeoArtifacts (oracle label marked oracle_only=True).
  3. Evaluate abs(delta NDVI) > threshold against OSCD change labels at
     multiple thresholds, computing Precision / Recall / F1 / IoU / Accuracy.
  4. Find the best threshold per city (by F1) and output global summary CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.datasets.oscd import (  # noqa: E402
    build_city_multispectral_geotiff,
    extract_zip_once,
    find_oscd_root,
    _find_label_path,
)
from geoharness.store import ArtifactStore  # noqa: E402
from geoharness.tools.evaluation import evaluate_change_mask, threshold_change_mask  # noqa: E402
from geoharness.tools.raster import raster_artifact  # noqa: E402

THRESHOLDS = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]


def _compute_ndvi_array(raster_path: Path) -> np.ndarray:
    """Compute NDVI from a multispectral GeoTIFF with red/nir band descriptions."""
    with rasterio.open(raster_path) as ds:
        bands = list(ds.descriptions)
        if "nir" not in bands or "red" not in bands:
            raise ValueError(f"Expected red and nir in band descriptions, got {bands}")
        nir = ds.read(bands.index("nir") + 1).astype("float32")
        red = ds.read(bands.index("red") + 1).astype("float32")
    denominator = nir + red
    return np.divide(
        nir - red, denominator,
        out=np.full_like(nir, np.nan),
        where=np.abs(denominator) > 1e-6,
    )


def _write_raster(data: np.ndarray, ref_path: Path, output_path: Path, band_name: str = "ndvi") -> None:
    """Write a single-band float32 raster using the profile of *ref_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSCD delta NDVI threshold evaluation.")
    parser.add_argument("--raw-dir", default="data/oscd/raw")
    parser.add_argument("--extract-dir", default="data/oscd/extracted")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    runs_dir = Path(args.runs_dir)

    # Ensure OSCD data is extracted
    for zip_path in sorted(raw_dir.glob("*.zip")):
        extract_zip_once(zip_path, extract_dir)
    labels_root = find_oscd_root(extract_dir, "Labels")
    cities = sorted(
        path.name for path in labels_root.iterdir() if path.is_dir()
    )
    if not cities:
        raise RuntimeError(f"No OSCD label cities found under {labels_root}")

    all_rows: list[dict] = []
    best_rows: list[dict] = []

    for city in cities:
        print(f"\n{'='*60}\n  {city}\n{'='*60}")
        city_workdir = runs_dir / f"oscd_{city}"
        eval_dir = city_workdir / "change_eval"
        store = ArtifactStore(eval_dir / "store")

        # Locate or build before/after GeoTIFFs
        before_path = city_workdir / "inputs" / f"{city}_before_rect.tif"
        after_path = city_workdir / "inputs" / f"{city}_after_rect.tif"

        if not before_path.exists() or not after_path.exists():
            # Build them on-the-fly
            images_root = find_oscd_root(extract_dir, "Images")
            before_path = build_city_multispectral_geotiff(
                images_root, city, 1,
                city_workdir / "inputs" / f"{city}_before_rect.tif",
            )
            after_path = build_city_multispectral_geotiff(
                images_root, city, 2,
                city_workdir / "inputs" / f"{city}_after_rect.tif",
            )

        # Compute NDVI arrays
        before_ndvi = _compute_ndvi_array(before_path)
        after_ndvi = _compute_ndvi_array(after_path)
        delta_ndvi = after_ndvi - before_ndvi

        # Save NDVI and delta rasters
        before_ndvi_path = eval_dir / "before_ndvi.tif"
        after_ndvi_path = eval_dir / "after_ndvi.tif"
        delta_path = eval_dir / "delta_ndvi.tif"
        _write_raster(before_ndvi, before_path, before_ndvi_path, "ndvi")
        _write_raster(after_ndvi, after_path, after_ndvi_path, "ndvi")
        _write_raster(delta_ndvi, after_path, delta_path, "delta_ndvi")

        # Register NDVI and delta as artifacts
        store.add(raster_artifact(
            "before_ndvi", before_ndvi_path,
            provenance={"tool": "ComputeIndex", "index": "NDVI", "role": "before"},
            bands=["ndvi"],
        ))
        store.add(raster_artifact(
            "after_ndvi", after_ndvi_path,
            provenance={"tool": "ComputeIndex", "index": "NDVI", "role": "after"},
            bands=["ndvi"],
        ))
        store.add(raster_artifact(
            "delta_ndvi", delta_path,
            parents=["before_ndvi", "after_ndvi"],
            provenance={"tool": "ComputeDelta", "operation": "after - before"},
            bands=["delta_ndvi"],
        ))

        # Register oracle label as artifact (hidden from agent)
        label_path = _find_label_path(labels_root, city)
        oracle_artifact = raster_artifact(
            "oracle_change_label", label_path,
            provenance={"tool": "OSCDAdapter", "role": "hidden_change_label"},
            bands=["change_label"],
        )
        oracle_artifact.metadata["oracle_only"] = True
        oracle_artifact.metadata["evaluation_context"] = "pixel_aligned_oracle"
        oracle_artifact.metadata["metric_space"] = "pixel_grid"
        oracle_artifact.metadata["crs_required_for_metric"] = False
        oracle_artifact.metadata["geospatial_deliverable"] = False
        store.add(oracle_artifact)

        # Evaluate each threshold
        city_rows = []
        for t in THRESHOLDS:
            mask_id = f"predicted_change_t{t:.2f}".replace(".", "_")
            metrics_id = f"change_eval_metrics_t{t:.2f}".replace(".", "_")

            # Create predicted change mask
            threshold_change_mask(store, "delta_ndvi", mask_id, threshold=t)

            # Evaluate against oracle
            evaluate_change_mask(store, mask_id, "oracle_change_label", metrics_id, threshold=t)

            metrics_artifact = store.get(metrics_id)
            df = pd.read_csv(metrics_artifact.path)
            row = df.iloc[0].to_dict()
            row["city"] = city
            row["threshold"] = t
            city_rows.append(row)
            all_rows.append(row)

        # Best threshold by F1
        city_frame = pd.DataFrame(city_rows)
        best_idx = city_frame["f1"].idxmax()
        best = city_frame.iloc[best_idx].to_dict()
        best_rows.append({
            "city": city,
            "best_threshold_by_f1": best["threshold"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "best_f1": best["f1"],
            "best_iou": best["iou"],
            "best_accuracy": best["accuracy"],
            "changed_pixels": int(best["true_changed_pixels"]),
            "predicted_changed_at_best": int(best["predicted_changed_pixels"]),
        })

        # Write per-city best threshold
        best_path = eval_dir / "best_threshold.json"
        best_path.write_text(json.dumps(best_rows[-1], indent=2, ensure_ascii=False), encoding="utf-8")

        # Write per-city threshold sweep
        per_city_csv = eval_dir / "threshold_metrics.csv"
        city_frame.to_csv(per_city_csv, index=False)

        # Write per-city evaluation report
        report_lines = [
            f"# OSCD Change Detection Evaluation — {city}",
            "",
            f"## Best Threshold (by F1)",
            "",
            f"- Threshold: {best['threshold']:.3f}",
            f"- Precision: {best['precision']:.4f}",
            f"- Recall:    {best['recall']:.4f}",
            f"- F1:        {best['f1']:.4f}",
            f"- IoU:       {best['iou']:.4f}",
            f"- Accuracy:  {best['accuracy']:.4f}",
            "",
            f"## Threshold Sweep",
            "",
            city_frame.to_markdown(index=False),
            "",
            "This evaluation uses a frozen heuristic backend (abs(delta NDVI) > threshold).",
            "The OSCD label is oracle-only and not visible to any agent or workflow planning step.",
            "",
            "## Evaluation Context",
            "",
            "This evaluation is **pixel-level**. The predicted mask and OSCD oracle label are",
            "compared on the aligned pixel grid. CRS diagnostics (e.g. `missing_crs`) on OSCD",
            "rect images indicate limited geospatial deliverability — they do **not** invalidate",
            "pixel-wise metrics (Precision/Recall/F1/IoU/Accuracy).",
            "",
            "本评估是像素级 oracle evaluation。预测 mask 和 OSCD label 在对齐像素网格上比较。",
            "CRS diagnostic 表示该产物的地理交付能力有限，并不表示 F1/IoU 等像素级指标无效。",
        ]
        report_path = eval_dir / "change_eval_report.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"  best threshold = {best['threshold']:.2f}, F1 = {best['f1']:.4f}, IoU = {best['iou']:.4f}")

    # ── Global summaries ──────────────────────────────────────────────

    all_frame = pd.DataFrame(all_rows)
    all_csv = runs_dir / "oscd_threshold_eval.csv"
    all_md = runs_dir / "oscd_threshold_eval.md"
    all_frame.to_csv(all_csv, index=False)
    all_md.write_text(all_frame.to_markdown(index=False), encoding="utf-8")
    print(f"\nwrote {all_csv} ({len(all_frame)} rows)")

    best_frame = pd.DataFrame(best_rows)
    best_csv = runs_dir / "oscd_best_thresholds.csv"
    best_md = runs_dir / "oscd_best_thresholds.md"
    best_frame.to_csv(best_csv, index=False)
    best_md.write_text(best_frame.to_markdown(index=False), encoding="utf-8")
    print(f"wrote {best_csv}")
    print(best_frame.to_markdown(index=False))

    # Print summary stats
    avg_f1 = best_frame["best_f1"].mean()
    avg_iou = best_frame["best_iou"].mean()
    avg_thresh = best_frame["best_threshold_by_f1"].mean()
    print(f"\nMean best F1 across {len(cities)} cities: {avg_f1:.4f}")
    print(f"Mean best IoU: {avg_iou:.4f}")
    print(f"Mean best threshold: {avg_thresh:.3f}")


if __name__ == "__main__":
    main()
