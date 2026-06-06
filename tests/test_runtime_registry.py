from __future__ import annotations

import pytest

from geoharness.registry import ToolRegistry
from geoharness.runtime import GeoHarnessRuntime
from geoharness.schemas import GeoArtifact, GeoSkillCall, GeoSkillResult, GeoSkillSpec
from geoharness.store import ArtifactStore


def _spec(name: str = "EchoTool") -> GeoSkillSpec:
    return GeoSkillSpec(
        name=name,
        family="test",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        preconditions=[],
        diagnostics=[],
    )


def test_registered_tool_executes_successfully(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    registry = ToolRegistry()

    def handler(store: ArtifactStore, artifact_id: str) -> GeoSkillResult:
        artifact = GeoArtifact(
            id=artifact_id,
            type="json",
            path=str(store.artifact_path(artifact_id, ".json")),
        )
        store.add(artifact)
        return GeoSkillResult(status="success", artifacts=[artifact])

    registry.register(_spec(), handler)
    registry.register(_spec("AlphaTool"), handler)
    runtime = GeoHarnessRuntime(store, registry)

    result = runtime.execute(GeoSkillCall(tool_name="EchoTool", arguments={"artifact_id": "echo"}))

    assert result.status == "success"
    assert result.artifacts[0].id == "echo"
    assert store.get("echo").id == "echo"
    assert [spec.name for spec in registry.list_specs()] == ["AlphaTool", "EchoTool"]


def test_unknown_tool_returns_fatal_diagnostic(tmp_path) -> None:
    runtime = GeoHarnessRuntime(ArtifactStore(tmp_path), ToolRegistry())

    result = runtime.execute(GeoSkillCall(tool_name="MissingTool", arguments={}))

    assert result.status == "failed"
    assert result.diagnostics[0].code == "unknown_tool"
    assert result.diagnostics[0].severity == "fatal"


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()

    def handler(store: ArtifactStore) -> GeoSkillResult:
        return GeoSkillResult(status="success")

    registry.register(_spec(), handler)

    with pytest.raises(ValueError, match="tool already registered: EchoTool"):
        registry.register(_spec(), handler)


def test_handler_exception_returns_fatal_diagnostic(tmp_path) -> None:
    registry = ToolRegistry()

    def handler(store: ArtifactStore) -> GeoSkillResult:
        raise RuntimeError("boom")

    registry.register(_spec("ExplodingTool"), handler)
    runtime = GeoHarnessRuntime(ArtifactStore(tmp_path), registry)

    result = runtime.execute(GeoSkillCall(tool_name="ExplodingTool", arguments={}))

    assert result.status == "failed"
    assert result.diagnostics[0].code == "tool_execution_error"
    assert result.diagnostics[0].severity == "fatal"
    assert result.diagnostics[0].message == "boom"
