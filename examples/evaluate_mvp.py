from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoharness.eval.metrics import (  # noqa: E402
    artifact_validity_rate,
    diagnostic_recall_on_injected_failures,
    provenance_completeness,
)
from geoharness.schemas import Diagnostic, GeoArtifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GeoHarness MVP summary file.")
    parser.add_argument("summary", nargs="?", default="runs/measure_mvp/summary.json")
    parser.add_argument("--expected-code", action="append", default=[])
    args = parser.parse_args()

    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    artifacts = [GeoArtifact(**item) for item in payload.get("artifacts", [])]
    diagnostics = [Diagnostic(**item) for item in payload.get("diagnostics", [])]
    metrics = {
        "workflow_success": payload.get("status") in {"success", "warning"},
        "artifact_validity_rate": artifact_validity_rate(artifacts, diagnostics),
        "diagnostic_recall_on_injected_failures": diagnostic_recall_on_injected_failures(
            set(args.expected_code),
            diagnostics,
        ),
        "provenance_completeness": provenance_completeness(artifacts),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
