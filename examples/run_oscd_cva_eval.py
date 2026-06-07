"""OSCD CVA (Change Vector Analysis) baseline — multi-band spectral change.

For each OSCD test-label city:
  1. Compute CVA score = sqrt(sum((band_after_i - band_before_i)^2))
  2. Evaluate percentile thresholds against OSCD labels.
  3. Register CVA score, predicted masks, and evaluation metrics as GeoArtifacts.
  4. Output per-city and global summaries, with delta NDVI vs CVA comparison.
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

# Percentile thresholds — top K% changed
PERCENTILES = [99, 98, 95, 90, 85, 80]  # top 1%, 2%, 5%, 10%, 15%, 20%


def _compute_score_percentile(score: np.ndarray, pct: int) -> float:
    """Compute the threshold that keeps top (100-pct)% of valid pixels as changed."""
    valid = score[np.isfinite(score)]
    if valid.size == 0:
        return 0.0
    return float(np.percentile(valid, pct))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSCD CVA threshold evaluation.")
    parser.add_argument("--raw-dir", default="data/oscd/raw")
    parser.add_argument("--extract-dir", default="data/oscd/extracted")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    runs_dir = Path(args.runs_dir)

    for zip_path in sorted(raw_dir.glob("*.zip")):
        extract_zip_once(zip_path, extract_dir)
    labels_root = find_oscd_root(extract_dir, "Labels")
    cities = sorted(path.name for path in labels_root.iterdir() if path.is_dir())
    if not cities:
        raise RuntimeError(f"No OSCD label cities found under {labels_root}")

    all_pct_rows: list[dict] = []
    best_pct_rows: list[dict] = []

    for city in cities:
        print(f"\n{'='*60}\n  CVA: {city}\n{'='*60}")
        city_workdir = runs_dir / f"oscd_{city}"
        cva_dir = city_workdir / "cva_eval"
        store = ArtifactStore(cva_dir / "store")

        # Locate before/after GeoTIFFs
        before_path = city_workdir / "inputs" / f"{city}_before_rect.tif"
        after_path = city_workdir / "inputs" / f"{city}_after_rect.tif"
        if not before_path.exists() or not after_path.exists():
            images_root = find_oscd_root(extract_dir, "Images")
            before_path = build_city_multispectral_geotiff(
                images_root, city, 1, city_workdir / "inputs" / f"{city}_before_rect.tif",
            )
            after_path = build_city_multispectral_geotiff(
                images_root, city, 2, city_workdir / "inputs" / f"{city}_after_rect.tif",
            )

        # Register before/after rasters (OSCD rect images lack CRS — fine for pixel-level CVA)
        before_artifact = raster_artifact(
            "cva_before", before_path,
            provenance={"tool": "OSCDAdapter", "role": "before"}, bands=["blue", "green", "red", "nir"],
        )
        after_artifact = raster_artifact(
            "cva_after", after_path,
            provenance={"tool": "OSCDAdapter", "role": "after"}, bands=["blue", "green", "red", "nir"],
        )
        store.add(before_artifact)
        store.add(after_artifact)

        # Compute CVA score directly (pixel-level, no CRS required)
        cva_path = cva_dir / "cva_score.tif"
        cva_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(before_path) as b_ds, rasterio.open(after_path) as a_ds:
            b_data = b_ds.read().astype("float32")
            a_data = a_ds.read().astype("float32")
            diff = a_data - b_data
            cva_data = np.sqrt(np.sum(diff ** 2, axis=0))
            # Mask nodata pixels
            for i in range(b_ds.count):
                if b_ds.nodata is not None and not np.isnan(b_ds.nodata):
                    cva_data[b_data[i] == b_ds.nodata] = np.nan
                if a_ds.nodata is not None and not np.isnan(a_ds.nodata):
                    cva_data[a_data[i] == a_ds.nodata] = np.nan
            cva_data[~np.isfinite(cva_data)] = np.nan
            profile = a_ds.profile.copy()
            profile.update(count=1, dtype="float32", nodata=np.nan)
            with rasterio.open(cva_path, "w", **profile) as dst:
                dst.write(cva_data, 1)

        cva_artifact = raster_artifact(
            "cva_score", cva_path,
            parents=["cva_before", "cva_after"],
            provenance={"tool": "ComputeCVA", "band_count": b_ds.count},
            bands=["cva_score"],
        )
        cva_artifact.quality["valid_pixels"] = int(np.sum(np.isfinite(cva_data)))
        store.add(cva_artifact)

        # Register oracle label
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

        # ── Percentile thresholds ──────────────────────────────────
        city_pct_rows = []
        for pct in PERCENTILES:
            pct_threshold = _compute_score_percentile(cva_data, pct)
            mask_id = f"cva_pred_change_pct_{pct}".replace(".", "_")
            metrics_id = f"cva_eval_pct_{pct}".replace(".", "_")
            threshold_change_mask(store, "cva_score", mask_id, threshold=pct_threshold)
            evaluate_change_mask(store, mask_id, "oracle_change_label", metrics_id, threshold=pct_threshold)
            row = pd.read_csv(store.get(metrics_id).path).iloc[0].to_dict()
            row["city"] = city
            row["threshold_type"] = "percentile"
            row["percentile"] = pct
            row["threshold"] = pct_threshold
            city_pct_rows.append(row)
            all_pct_rows.append(row)

        # Best percentile by F1
        pct_df = pd.DataFrame(city_pct_rows)
        best_pct = pct_df.iloc[pct_df["f1"].idxmax()].to_dict()
        best_pct_rows.append({"city": city, "threshold_type": "percentile",
            "best_percentile": best_pct["percentile"], "best_threshold": best_pct["threshold"],
            "best_f1": best_pct["f1"], "best_iou": best_pct["iou"],
            "best_precision": best_pct["precision"], "best_recall": best_pct["recall"],
            "best_accuracy": best_pct["accuracy"]})
        print(f"  pct best: pct={best_pct['percentile']}, thr={best_pct['threshold']:.4f}, F1={best_pct['f1']:.4f}, IoU={best_pct['iou']:.4f}")

        # Write per-city summary
        per_city_csv = cva_dir / "threshold_metrics.csv"
        pct_df.to_csv(per_city_csv, index=False)
        best_path = cva_dir / "best_threshold.json"
        best_path.write_text(json.dumps(best_pct_rows[-1], indent=2, ensure_ascii=False), encoding="utf-8")

        # Per-city CVA eval report
        report_lines = [
            f"# OSCD CVA Change Detection Evaluation — {city}",
            "",
            "## Best Percentile Threshold",
            f"- Percentile: {best_pct['percentile']} (top {100-best_pct['percentile']}%)",
            f"- Threshold: {best_pct['threshold']:.4f}",
            f"- Precision: {best_pct['precision']:.4f}",
            f"- Recall: {best_pct['recall']:.4f}",
            f"- F1: {best_pct['f1']:.4f}",
            f"- IoU: {best_pct['iou']:.4f}",
            f"- Accuracy: {best_pct['accuracy']:.4f}",
            "",
            "## Percentile Threshold Sweep",
            pct_df[["percentile", "threshold", "precision", "recall", "f1", "iou", "accuracy"]].to_markdown(index=False),
            "",
            "## Evaluation Context",
            "",
            "This evaluation is **pixel-level**. The predicted mask and OSCD oracle label are",
            "compared on the aligned pixel grid. CRS diagnostics (e.g. `missing_crs`) on OSCD",
            "rect images indicate limited geospatial deliverability — they do **not** invalidate",
            "pixel-wise metrics.",
            "",
            "本评估是像素级 oracle evaluation。CVA score 基于所有可用波段计算，",
            "比单一 delta NDVI 更适合多光谱城市变化检测。",
        ]
        report_path = cva_dir / "cva_eval_report.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # ── Global summaries ────────────────────────────────────────────

    # Percentile thresholds
    pct_all = pd.DataFrame(all_pct_rows)
    pct_csv = runs_dir / "oscd_cva_pct_eval.csv"
    pct_md = runs_dir / "oscd_cva_pct_eval.md"
    pct_all.to_csv(pct_csv, index=False)
    pct_md.write_text(pct_all.to_markdown(index=False), encoding="utf-8")
    print(f"wrote {pct_csv}")

    # Best thresholds
    best_pct_df = pd.DataFrame(best_pct_rows)
    best_pct_df.to_csv(runs_dir / "oscd_cva_best_pct.csv", index=False)
    print("wrote oscd_cva_best_pct.csv")

    # ── Comparison with delta NDVI baseline ─────────────────────────

    # Read delta NDVI best thresholds for comparison
    ndvi_best_path = runs_dir / "oscd_best_thresholds.csv"
    ndvi_best = pd.read_csv(ndvi_best_path) if ndvi_best_path.exists() else None

    comparison_rows = []
    for _, row in best_pct_df.iterrows():
        comparison_rows.append({
            "city": row["city"], "baseline": "CVA (percentile)",
            "best_threshold": f"pct={row['best_percentile']}",
            "best_f1": row["best_f1"], "best_iou": row["best_iou"],
        })
    if ndvi_best is not None:
        for _, row in ndvi_best.iterrows():
            comparison_rows.append({
                "city": row["city"], "baseline": "delta NDVI (fixed)",
                "best_threshold": row["best_threshold_by_f1"],
                "best_f1": row["best_f1"], "best_iou": row["best_iou"],
            })

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = runs_dir / "oscd_baseline_comparison.csv"
    comp_md = runs_dir / "oscd_baseline_comparison.md"
    comp_df.to_csv(comp_csv, index=False)
    comp_md.write_text(comp_df.to_markdown(index=False), encoding="utf-8")
    print(f"wrote {comp_csv}")

    # ── Mean comparison ────────────────────────────────────────────
    print("\n=== Baseline Comparison (Mean across 10 cities) ===")
    for label, df, f1_col, iou_col in [
        ("delta NDVI (fixed)", ndvi_best, "best_f1", "best_iou"),
        ("CVA (percentile)", best_pct_df, "best_f1", "best_iou"),
    ]:
        if df is not None and not df.empty:
            print(f"  {label}: mean F1={df[f1_col].mean():.4f}, mean IoU={df[iou_col].mean():.4f}")


if __name__ == "__main__":
    main()
