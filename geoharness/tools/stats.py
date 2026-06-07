from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from geoharness.feedback import validate_raster_artifact
from geoharness.schemas import Diagnostic, GeoArtifact, GeoSkillResult, status_from_diagnostics
from geoharness.store import ArtifactStore


def zonal_statistics(
    store: ArtifactStore,
    raster_id: str,
    output_id: str,
) -> GeoSkillResult:
    source = store.get(raster_id)
    diagnostics = validate_raster_artifact(source)
    output_path = store.artifact_path(output_id, ".csv")

    with rasterio.open(source.path) as dataset:
        data = dataset.read(1, masked=True)
        valid_count = int(data.count())
        if valid_count == 0:
            diagnostics.append(
                Diagnostic(
                    code="empty_statistics_input",
                    severity="fatal",
                    message="Cannot compute statistics because the raster has no valid pixels.",
                    artifact_id=raster_id,
                    check_name="zonal_statistics",
                )
            )
            store.record_diagnostics(diagnostics)
            return GeoSkillResult(status="failed", diagnostics=diagnostics)
        values = data.compressed().astype("float64")

    frame = pd.DataFrame(
        [
            {
                "artifact_id": raster_id,
                "valid_pixels": valid_count,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }
        ]
    )
    frame.to_csv(output_path, index=False)

    artifact = GeoArtifact(
        id=output_id,
        type="table",
        path=str(output_path),
        parents=[raster_id],
        provenance={"tool": "ZonalStatistics", "input": raster_id},
        metadata={"columns": list(frame.columns), "rows": len(frame)},
    )
    store.add(artifact)
    store.record_diagnostics(diagnostics)
    return GeoSkillResult(
        status=status_from_diagnostics(diagnostics),
        artifacts=[artifact],
        diagnostics=diagnostics,
        provenance={"tool": "ZonalStatistics", "input": raster_id},
    )


def write_measure_report(
    store: ArtifactStore,
    table_id: str,
    output_id: str,
    *,
    title: str,
) -> GeoSkillResult:
    table = store.get(table_id)
    output_path = store.artifact_path(output_id, ".md")
    frame = pd.read_csv(table.path)
    row = frame.iloc[0].to_dict()
    text = (
        f"# {title}\n\n"
        f"- Source artifact: `{row['artifact_id']}`\n"
        f"- Valid pixels: {int(row['valid_pixels'])}\n"
        f"- Mean value: {row['mean']:.4f}\n"
        f"- Median value: {row['median']:.4f}\n"
        f"- Min / max: {row['min']:.4f} / {row['max']:.4f}\n\n"
        "This report is generated from the artifact graph metadata and CSV output.\n"
    )
    Path(output_path).write_text(text, encoding="utf-8")
    artifact = GeoArtifact(
        id=output_id,
        type="report",
        path=str(output_path),
        parents=[table_id],
        provenance={"tool": "WriteMeasureReport", "input": table_id},
    )
    store.add(artifact)
    return GeoSkillResult(status="success", artifacts=[artifact], provenance={"tool": "WriteMeasureReport"})
