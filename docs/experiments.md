# Experiments and reproduction

## Available experiment: Week 1 ingestion demo

**Purpose:** validate data ingestion, timestamp normalization, event-count conservation, and deterministic visualization. No motion or trajectory estimator is run.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
adaptive-event-vo demo --seed 7 --output-dir outputs/demo
```

The source stream is generated and saved, then reloaded using the same public loader as user recordings. The CLI prints the validated event count and artifact paths. `event_stream.json` stores the command, seed, dimensions, raw timestamp origin/unit, observed span, polarity totals, four window bounds/counts, source SHA-256, raster sample count, environment versions, and the sums used in both density plots.

Check that `positive_events + negative_events`, `spatial_count_sum`, and `temporal_count_sum` each equal `event_count`. Display windows cover only their stated intervals, so their counts need not sum to the recording total. The integration test checks these invariants automatically.

The README figure and summary are selected copies of this run's outputs:

```bash
python -c 'from pathlib import Path; import shutil; Path("results/figures").mkdir(parents=True, exist_ok=True); shutil.copyfile("outputs/demo/event_stream.png", "results/figures/event_stream.png"); shutil.copyfile("outputs/demo/event_stream.json", "results/ingestion_demo.json")'
```

The saved summary describes a local reference run, not a benchmark. Seeded arrays should reproduce with the same NumPy version; font rendering, PNG metadata, platform, and archive bytes may differ across environments. Source hashes identify a particular input artifact and are not a claim that figures or compressed archives are byte-identical across machines. The fixture-generation and round-trip tests compare event arrays.

For a version-matched reference run, use Python 3.12 and install `numpy==2.3.5` and `matplotlib==3.10.8` before installing this package. Transitive dependencies are not locked; the runtime versions are recorded in each summary. CI is configured for Python 3.10 and 3.12; only locally executed checks should be described as passing.

On a restricted host with an unwritable home directory, set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable locations before running plots, for example `/tmp/adaptive-event-vo-mpl` and `/tmp/adaptive-event-vo-cache`. On ordinary installations no extra environment configuration is required.

## Planned VO benchmark

A real-sequence **image-motion** baseline is available; no trajectory result or adaptive comparison is available yet. See [Week 2 reproduction](week2.md) for the real DAVIS clip, controlled synthetic displacement checks, timing scope and recorded evidence. The protocol below remains the planned VO experiment.

### Comparison and tuning

- Fixed nonoverlapping windows: 10, 25, and 50 ms.
- Fixed event count: choose the count on development sequences, then freeze it.
- Adaptive controller: tune budget, bounds, smoothing and activity weight only on development sequences.
- Ablations: density-only, motion-only, combined; optional representation changes in a separate study.

Use the same test sequence intervals, representation, feature/correspondence settings, pose solver, calibration, and failure policy. Publish both development and held-out sequence lists. Week 2 provides a complete fixed-window scheduler; its final partial-window and endpoint conventions must be matched by later policies. The Week 1 viewer's four diagnostic windows remain separate from the complete baseline schedule.

### Manifest required for each benchmark

| Category | Required record |
| --- | --- |
| Provenance | Repository revision, dirty state, full config, random seeds, dataset identifiers and file hashes |
| Sensor | Dimensions, intrinsics, distortion model, camera/body extrinsics, timestamp unit and synchronization |
| Data split | Development/test sequences, exact start/end intervals, ground-truth coverage and exclusions |
| Runtime | CPU/GPU, RAM, OS, Python/dependency versions, thread count and power settings |
| Accuracy | Pose convention, timestamp association/interpolation, alignment type and scale, ATE/RPE definition and lag |
| Tracking | Successful interval coverage, failed/invalid poses, termination/reinitialization policy |
| Efficiency | Timing boundaries, warm-up, repetition count, p50/p95 latency, total compute and accumulation delay |
| Workload | Input/processed/rejected events, repetitions from overlap, representation count and peak memory |

### Timing and reporting

Separate decode, representation, correspondence, pose estimation, and total processing time using a monotonic clock. Disable plotting during timed runs, use an identical I/O policy, and predeclare repetitions and warm-up. Report variation rather than only a best run. Do not equate event rate with camera motion or infer reduced work from duration alone.

Publish per-sequence tables alongside aggregates. Planned figures are aligned estimated versus ground-truth trajectories and accuracy versus total compute with tracking coverage and variability. Include failures and unfavorable sequences. Save machine-readable CSV/JSON under `results/tables/` only after results are real and reproducible.
