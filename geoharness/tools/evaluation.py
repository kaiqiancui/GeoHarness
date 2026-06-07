"""Evaluation tools — oracle comparison artefacts for change detection.

These tools produce thresholded change masks and compute standard
pixel-level metrics (precision, recall, F1, IoU, accuracy) against
oracle labels, registering every intermediate artefact in the store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

from geoharness.feedback import validate_raster_artifact
from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore
from geoharness.tools.raster import raster_artifact


def threshold_change_mask(
    store: ArtifactStore,
    delta_raster_id: str,
    output_id: str,
    *,
    threshold: float = 0.10,
) -> GeoSkillResult:
    """Binarize an absolute delta raster into a predicted change mask.

    ``abs(delta) > threshold`` → changed (1), otherwise unchanged (0).
    """
    source = store.get(delta_raster_id)
    diagnostics = validate_raster_artifact(source)

    output_path = store.artifact_path(output_id, ".tif")
    with rasterio.open(source.path) as src:
        data = src.read(1).astype("float32")
        profile = src.profile.copy()
        # Binary mask: 0 = unchanged/background (valid class), NOT nodata.
        # Strip source nodata (e.g. NaN from float32) so it doesn't conflict with uint8.
        profile.update(dtype="uint8", nodata=None)

        valid = np.isfinite(data)
        mask = np.where(valid, (np.abs(data) > threshold).astype("uint8"), 0)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

    valid_count = int(np.sum(valid))
    positive_count = int(np.sum(mask))
    positive_ratio = positive_count / valid_count if valid_count > 0 else 0.0

    artifact = raster_artifact(
        output_id,
        output_path,
        parents=[delta_raster_id],
        provenance={
            "tool": "ThresholdChangeMask",
            "input": delta_raster_id,
            "threshold": threshold,
            "mode": "abs_delta_greater_than",
        },
        bands=["predicted_change"],
    )
    artifact.quality["positive_pixel_ratio"] = positive_ratio
    artifact.quality["predicted_changed_pixels"] = positive_count
    artifact.quality["valid_pixels"] = valid_count
    artifact.metadata["artifact_role"] = "binary_mask"
    artifact.metadata["mask_encoding"] = {"background": 0, "positive": 1}
    artifact.metadata["evaluation_context"] = "pixel_aligned_oracle"
    artifact.metadata["metric_space"] = "pixel_grid"
    artifact.metadata["crs_required_for_metric"] = False
    artifact.metadata["geospatial_deliverable"] = False
    store.add(artifact)
    store.record_diagnostics(diagnostics)

    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "ThresholdChangeMask", "threshold": threshold},
    )


def evaluate_change_mask(
    store: ArtifactStore,
    predicted_id: str,
    oracle_id: str,
    output_id: str,
    *,
    threshold: float | None = None,
    label_encoding: dict[str, int] | None = None,
) -> GeoSkillResult:
    """Compute pixel-level classification metrics against an oracle label.

    Parameters
    ----------
    label_encoding : dict
        Mapping of semantic class to pixel value in the oracle raster.
        Default for OSCD: ``{"unchanged": 1, "changed": 2}``.
    """
    if label_encoding is None:
        label_encoding = {"unchanged": 1, "changed": 2}

    predicted_src = store.get(predicted_id)
    oracle_src = store.get(oracle_id)
    diagnostics: list[Diagnostic] = []

    diagnostics.extend(validate_raster_artifact(predicted_src))
    diagnostics.extend(validate_raster_artifact(oracle_src))

    with rasterio.open(predicted_src.path) as pred_ds, rasterio.open(oracle_src.path) as oracle_ds:
        pred = pred_ds.read(1).astype("uint8")
        oracle = oracle_ds.read(1)

        # Align shapes if needed (OSCD labels may differ in size)
        if oracle.shape != pred.shape:
            oracle = np.array(
                Image.fromarray(oracle).resize(
                    pred.shape[::-1], Image.Resampling.NEAREST
                )
            )

        # OSCD: 1 = unchanged, 2 = changed.  Predicted: 1 = changed, 0 = unchanged.
        # All other values in oracle are treated as ignore.
        true_changed = oracle == label_encoding["changed"]
        true_unchanged = oracle == label_encoding["unchanged"]
        valid_oracle = true_changed | true_unchanged

        pred_changed = pred == 1
        pred_unchanged = ~pred_changed  # includes nodata

        # Only evaluate pixels where oracle has a valid label AND pred has data
        eval_mask = valid_oracle

        tp = int(np.sum(pred_changed & true_changed & eval_mask))
        fp = int(np.sum(pred_changed & true_unchanged & eval_mask))
        fn = int(np.sum(pred_unchanged & true_changed & eval_mask))
        tn = int(np.sum(pred_unchanged & true_unchanged & eval_mask))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    output_path = store.artifact_path(output_id, ".csv")
    frame = pd.DataFrame([{
        "artifact_id": predicted_id,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "predicted_changed_pixels": int(tp + fp),
        "true_changed_pixels": int(tp + fn),
        "true_unchanged_pixels": int(tn + fp),
        "oracle_label_encoding": str(label_encoding),
    }])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    artifact = GeoArtifact(
        id=output_id,
        type="table",
        path=str(output_path),
        parents=[predicted_id, oracle_id],
        provenance={
            "tool": "EvaluateChangeMask",
            "prediction": predicted_id,
            "oracle": oracle_id,
            "threshold": threshold,
            "label_encoding": label_encoding,
            "metrics": ["precision", "recall", "f1", "iou", "accuracy"],
        },
        metadata={
            "columns": list(frame.columns), "rows": len(frame),
            "evaluation_context": "pixel_aligned_oracle",
            "metric_space": "pixel_grid",
            "crs_required_for_metric": False,
            "geospatial_deliverable": False,
        },
    )
    store.add(artifact)
    store.record_diagnostics(diagnostics)

    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "EvaluateChangeMask", "threshold": threshold},
    )
