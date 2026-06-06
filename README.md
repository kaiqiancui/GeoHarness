# GeoHarness

GeoHarness is an early research prototype for artifact-centric execution and feedback in remote-sensing agent workflows.

This repository currently contains a local MVP for a `Measure` workflow:

```text
synthetic GeoTIFF + AOI
-> load raster
-> clip by AOI
-> compute NDVI
-> compute statistics
-> write report
-> record artifact metadata and diagnostics
```

## Quick Start

Run inside the course conda environment:

```bash
conda run -n rsiip python examples/run_measure_mvp.py
```

The command writes outputs under `runs/measure_mvp/`, including:

- `summary.json`
- `store/metadata.json`
- `store/diagnostics.jsonl`
- GeoTIFF, CSV, and Markdown report artifacts

Run the injected-failure diagnostic smoke test:

```bash
conda run -n rsiip python examples/run_diagnostic_stress.py
```

Evaluate a generated summary:

```bash
conda run -n rsiip python examples/evaluate_mvp.py runs/measure_mvp/summary.json
```

Run tests:

```bash
conda run -n rsiip python -m pytest
```

## OSCD Real-Data Experiment

OSCD support is an experimental real-data workflow. It is not bundled with the
repo; manually download the OSCD images and test labels into `data/oscd/raw/`
before running it. This workflow depends on Pillow for label resizing.

```bash
conda run -n rsiip python examples/prepare_oscd.py
conda run -n rsiip python examples/run_oscd_experiment.py --city brasilia
conda run -n rsiip python examples/run_oscd_all.py
conda run -n rsiip python examples/summarize_oscd_results.py
conda run -n rsiip python examples/visualize_oscd.py --city brasilia
```

The OSCD experiment builds compact 4-band GeoTIFFs from one Sentinel-2 city pair,
runs the Measure workflow on the after image, and reports NDVI-change statistics
against the OSCD change label.
