"""GeoHarnessAgent — LLM-driven ReAct loop for geo-spatial workflows.

The agent translates a natural-language task into a sequence of GeoHarness
tool calls, executes them through the artifact store, observes structured
diagnostics, and iterates until the task is complete.

Usage::

    from geoharness.agent import GeoHarnessAgent
    from geoharness.llm.client import create_client

    client = create_client("deepseek")
    agent = GeoHarnessAgent(client, store_root="runs/agent_demo")
    result = agent.run("Compute NDVI for the synthetic scene and tell me the mean value.")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geoharness.llm.client import LLMClient, LLMResponse, ToolCall
from geoharness.llm.context import (
    ToolCallRecord,
    format_final_summary,
    format_step_result,
    format_tool_result_for_llm,
)
from geoharness.llm.tools import GEOHARNESS_TOOLS, get_handler, llm_tool_schemas
from geoharness.store import ArtifactStore

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a GeoHarness agent — an AI assistant that executes remote-sensing and \
geospatial analysis workflows through structured tool calls.

## How GeoHarness Works

Every operation produces a **GeoArtifact** — a structured record that carries:
- Spatial metadata: CRS, bounds, resolution, shape, band names
- Provenance: which parent artifacts it was derived from
- Quality: valid-pixel ratio, positive-pixel ratio, etc.

Artifacts are identified by **string IDs** that you choose (e.g. "raw_scene", \
"clipped_scene", "ndvi_raster"). You chain tools by passing artifact IDs \
between them.

After each tool call you will receive:
- **status**: "success", "warning", or "failed"
- **artifacts**: list of created artifacts with their metadata
- **diagnostics**: structured issue codes with severity and suggested actions
- **warnings**: human-readable warning messages

## Diagnostics

The diagnostic engine reports issues with standardized codes:
- **fatal**: the step failed, you cannot use this artifact (e.g. "missing_crs", \
"aoi_outside_raster", "missing_band", "file_not_found")
- **warning**: the step succeeded but there are quality concerns (e.g. \
"low_valid_pixel_ratio", "unsafe_geographic_crs", "empty_mask")

When you see a **fatal** diagnostic, you must either:
1. Fix the input and retry (e.g. use a different AOI, check file paths)
2. Acknowledge the task cannot be completed and explain why

When you see a **warning**, you can continue but should note the risk in your \
final answer.

## Standard Workflows

**Measure** (single image → statistics):
  1. load_raster → 2. load_vector (AOI) → 3. clip_by_aoi → \
4. compute_index → 5. zonal_statistics → 6. write_report

**Detect** (single image → binary mask → vector regions):
  1-4 as Measure → 5. threshold_mask → 6. vectorize_mask → \
7. mask_area_statistics → 8. write_report

**Compare** (before/after change detection):
  1. load_raster (before) → 2. load_raster (after) → 3. load_vector (AOI) → \
4. validate_raster_pair → 5-6. clip both → 7-8. compute_index both → \
9. compute_delta → 10. change_statistics → 11. write_report

**CVA Change Detection** (multi-band spectral change):
  1-4 as Compare → 5. compute_cva_score → 6. threshold_change_mask → \
7. evaluate_change_mask → 8. write_report

## Available Indices

- **NDVI**: Normalised Difference Vegetation Index — requires nir + red bands
- **NDWI**: Normalised Difference Water Index — requires green + nir bands
- **NDBI**: Normalised Difference Built-up Index — requires swir + nir bands
- **NBR**: Normalised Burn Ratio — requires nir + swir bands

## Guidelines

1. **Plan first**: Before calling any tool, think about the workflow you need.
2. **One step at a time**: Each tool call depends on artifacts from previous steps.
3. **Use descriptive artifact IDs**: e.g. "before_scene", "clipped_scene", \
"ndvi_raster", "delta_ndvi".
4. **Check status after each call**: If a step fails, diagnose and decide before \
continuing.
5. **Provide a clear final answer**: Summarize what you did, key findings, and \
any warnings. Include specific numeric values when available.
6. **Be precise with paths**: Use absolute paths or paths relative to the \
working directory."""


# ── Agent ───────────────────────────────────────────────────────────────────────


class GeoHarnessAgent:
    """LLM-driven agent that executes GeoHarness workflows via function calling.

    Parameters
    ----------
    client : LLMClient
        The LLM backend (OpenAI, Anthropic, or DeepSeek).
    store_root : str or Path
        Directory for the artifact store.
    max_steps : int
        Maximum tool-call iterations before stopping (default 20).
    verbose : bool
        Print execution trace to stdout.
    """

    def __init__(
        self,
        client: LLMClient,
        store_root: str | Path = "runs/agent",
        max_steps: int = 20,
        verbose: bool = True,
        diagnostics_visible: bool = True,
        custom_tools: list[dict[str, Any]] | None = None,
        custom_handlers: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._store = ArtifactStore(store_root)
        self._max_steps = max_steps
        self._verbose = verbose
        self._diagnostics_visible = diagnostics_visible
        self._custom_tools = custom_tools
        self._custom_handlers = custom_handlers or {}
        self._trace: list[ToolCallRecord] = []

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def store(self) -> ArtifactStore:
        return self._store

    @property
    def trace(self) -> list[ToolCallRecord]:
        return self._trace

    def run(self, task: str) -> dict[str, Any]:
        """Execute a natural-language geo-spatial task.

        Returns a dict with ``status``, ``answer``, ``steps``, ``trace``,
        and ``metadata_path``.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        tools_schema = self._custom_tools if self._custom_tools is not None else llm_tool_schemas()
        final_answer: str | None = None

        for step in range(1, self._max_steps + 1):
            response = self._client.chat(messages, tools=tools_schema)

            if response.has_tool_calls:
                # ── execute tool calls ────────────────────────────────
                messages.append(_assistant_message(response))
                for tc in response.tool_calls:
                    result_msg = self._execute_and_record(step, tc)
                    messages.append(result_msg)
            else:
                # ── final answer ──────────────────────────────────────
                final_answer = response.content or ""
                break

        if final_answer is None:
            final_answer = (
                f"Reached maximum steps ({self._max_steps}) without a final answer. "
                "The task may be too complex or the agent may be stuck."
            )

        if self._verbose:
            print(format_final_summary(self._trace))
            print(f"\n{'=' * 60}")
            print(f"FINAL ANSWER:\n{final_answer}")
            print(f"{'=' * 60}")

        return {
            "status": "success" if len(self._trace) < self._max_steps else "max_steps",
            "answer": final_answer,
            "steps": len(self._trace),
            "trace": [self._trace_record_to_dict(r) for r in self._trace],
            "metadata_path": str(self._store.metadata_path),
        }

    # ── Internals ──────────────────────────────────────────────────────────

    def _execute_and_record(self, step: int, tc: ToolCall) -> dict[str, Any]:
        """Execute a single tool call, record the trace, and return the
        LLM-facing result message."""
        if self._verbose:
            print(f"\n--- Step {step}: {tc.name} ---")
            print(f"    args: {json.dumps(tc.arguments, ensure_ascii=False)}")

        # Use custom handler if available, otherwise look up GeoHarness handler
        handler = self._custom_handlers.get(tc.name) if self._custom_handlers else None
        if handler is None:
            handler = get_handler(tc.name)

        if handler is None:
            result = {
                "status": "failed",
                "artifacts": [],
                "diagnostics": [
                    {
                        "code": "unknown_tool",
                        "severity": "fatal",
                        "message": f"Unknown tool: {tc.name}",
                        "suggested_actions": ["check_tool_name"],
                    }
                ],
                "warnings": [],
            }
        else:
            try:
                raw = handler(self._store, **tc.arguments)
                if hasattr(raw, "to_dict"):
                    result = raw.to_dict()
                else:
                    result = raw
            except Exception as exc:
                import traceback
                tb_lines = traceback.format_exc()
                if self._verbose:
                    print(f"    [ERROR] {type(exc).__name__}: {exc}")
                    print(f"    {tb_lines[-3:]}")  # last 3 lines of traceback
                result = {
                    "status": "failed",
                    "artifacts": [],
                    "diagnostics": [
                        {
                            "code": "tool_execution_error",
                            "severity": "fatal",
                            "message": f"{type(exc).__name__}: {exc}",
                            "suggested_actions": [],
                        }
                    ],
                    "warnings": [],
                }

        # Strip diagnostics if not visible (ablation setting B)
        if not self._diagnostics_visible:
            result = {**result, "diagnostics": []}

        record = format_step_result(step, tc.name, tc.arguments, result)
        self._trace.append(record)

        if self._verbose:
            codes = [d["code"] for d in result.get("diagnostics", [])]
            print(f"    status: {result.get('status')}, diagnostics: {codes}")

        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": format_tool_result_for_llm(tc.name, result),
        }

    @staticmethod
    def _trace_record_to_dict(record: ToolCallRecord) -> dict[str, Any]:
        return {
            "step": record.step,
            "tool_name": record.tool_name,
            "arguments": record.arguments,
            "status": record.status,
            "diagnostic_codes": record.diagnostic_codes,
            "warnings": record.warnings,
            "artifact_ids": [a.id for a in record.artifacts],
        }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _assistant_message(response: LLMResponse) -> dict[str, Any]:
    """Build an assistant message with optional tool calls (OpenAI format)."""
    msg: dict[str, Any] = {"role": "assistant"}
    if response.content:
        msg["content"] = response.content
    if response.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in response.tool_calls
        ]
    return msg
