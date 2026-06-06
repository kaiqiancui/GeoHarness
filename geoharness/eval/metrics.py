from __future__ import annotations

from geoharness.schemas import Diagnostic, GeoArtifact


def artifact_validity_rate(artifacts: list[GeoArtifact], diagnostics: list[Diagnostic]) -> float:
    if not artifacts:
        return 0.0
    fatal_artifact_ids = {
        diagnostic.artifact_id
        for diagnostic in diagnostics
        if diagnostic.severity == "fatal" and diagnostic.artifact_id is not None
    }
    valid_count = sum(1 for artifact in artifacts if artifact.id not in fatal_artifact_ids)
    return valid_count / len(artifacts)


def diagnostic_recall_on_injected_failures(
    expected_codes: set[str],
    diagnostics: list[Diagnostic],
) -> float:
    if not expected_codes:
        return 1.0
    observed = {diagnostic.code for diagnostic in diagnostics}
    return len(expected_codes & observed) / len(expected_codes)


def provenance_completeness(artifacts: list[GeoArtifact]) -> float:
    if not artifacts:
        return 0.0
    complete = sum(1 for artifact in artifacts if artifact.provenance)
    return complete / len(artifacts)
