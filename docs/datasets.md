# Event recordings and data contract

## Supported formats

Both supported formats require four one-dimensional numeric columns/arrays with equal lengths. The recording must be nonempty.

| Field | Meaning | Accepted values |
| --- | --- | --- |
| `t` | Timestamp in the declared input unit | Finite, nonnegative, nondecreasing; duplicates allowed |
| `x` | Horizontal pixel coordinate | Integer in `[0, width)` |
| `y` | Vertical pixel coordinate | Integer in `[0, height)` |
| `p` | Contrast-change polarity | Binary `0/1` or signed `-1/+1`; encodings cannot be mixed |

Image coordinates have their origin at the top-left: x increases right and y increases down. The caller must supply sensor width and height; active-pixel extents are not reliable sensor dimensions.

**CSV:** a header containing `t,x,y,p` is required; column order may vary and additional named columns are ignored. Duplicate headers, missing fields, and rows with extra values are rejected. UTF-8 (including BOM) is accepted. For example, timestamps here are microseconds:

```csv
t,x,y,p
1000000,12,8,1
1000025,12,8,0
1000025,13,8,1
```

**NPZ:** use named numeric arrays `t`, `x`, `y`, `p`. Other arrays are ignored. Object arrays are unsupported; `allow_pickle=False` is always used. Metadata is deliberately not inferred from arbitrary archive fields.

```python
import numpy as np

np.savez_compressed(
    "events.npz",
    t=np.array([1_000_000, 1_000_025, 1_000_025], dtype=np.int64),
    x=np.array([12, 12, 13]),
    y=np.array([8, 8, 8]),
    p=np.array([1, 0, 1]),
)
```

## Timestamp policy

Units are mandatory: `s`, `ms`, `us`, or `ns`. Loading subtracts the first input timestamp, then converts to float64 seconds. The original first timestamp and unit are retained in `EventStream` and the run summary. Values exposed as `events.t` and passed to `events.window()` are always seconds relative to that recording origin.

For integer input, subtraction occurs **before** conversion to floating point. This preserves small differences between large epoch timestamps, such as nanoseconds above `2**53`. CSV integer timestamp tokens are parsed as int64 (out-of-range values are rejected); NPZ integer arrays retain their native dtype. Store large epoch timestamps as integers, not floats or scientific-notation CSV tokens: precision already lost in a floating input cannot be recovered.

Invalid ordering is an error. The loader does not silently sort, remove, deduplicate, clip, or denoise sensor events. Duplicate timestamps and repeated events at the same pixel are retained. Binary `0` maps to signed `-1`; `1` remains `+1`.

## Window and representation semantics

`events.window(start, end)` selects `[start, end)` using binary search. It preserves the recording's time origin; timestamps are not reset within each window. An event exactly on a shared boundary belongs to the later window. Empty windows are valid and render as white occupancy images. Events exactly at the last displayed window's right boundary are excluded from those four windows; full-recording diagnostics still include them.

The event-count representation has shape `(2, height, width)`, with negative polarity in channel 0 and positive in channel 1. Repeated events are added using integer accumulation. The RGB image encodes occupancy, not magnitude: blue is positive only, orange negative only, purple both, white neither. Spatial density sums both channels and displays `log(1 + count)`.

The temporal-rate panel uses 40 equal bins over the observed relative timestamp span. Rates are counts divided by bin width; the final histogram bin includes its right edge. If every event has the same timestamp, a 1 microsecond plotting span is used solely to make a drawable histogram. Observed timestamp span is not a guaranteed recording exposure time.

## Small example and synthetic provenance

`examples/synthetic_events.csv` contains 320 evenly indexed events selected from `generate_events(seed=7, width=32, height=24)`. It is generated locally, has no external license dependency, and can be regenerated with:

```bash
python scripts/generate_example.py
```

The full demo uses a 128×96 sensor grid. Its generator creates two moving vertical edges with opposing polarity, a varying event count over 200 time intervals, and uniform background noise. Randomness uses NumPy's seeded generator. Timestamps are quantized to microseconds and stably sorted during **fixture generation**, before loader validation. `save_events` writes CSV/NPZ with integer microsecond timestamps and rounds to the nearest microsecond; it does not preserve arbitrary input units or sub-microsecond detail.

Neither fixture models intensity-threshold triggering, calibrated camera motion, scene geometry, or sensor noise physics. They carry no pose ground truth and cannot support ATE/RPE experiments.

## External recordings: next milestone

The intended first real-data source is the DAVIS Event-Camera Dataset and Simulator (Mueggler et al., IJRR 2017). A specific sequence set, distribution terms, calibration, ground-truth coverage, and timestamp synchronization must be checked and recorded before evaluation. No external dataset has been downloaded or evaluated in this release.

Convert source events explicitly to the schema above; keep original data and conversion outputs under `data/`. Document the source URL, sequence, file hash, conversion command, timestamp unit, sensor size, polarity mapping, calibration, and any preprocessing. Raw text with no header, ROS bag, HDF5, and AEDAT ingestion are not implemented. There is no download script claiming to support them.

The current loader and report load the whole recording into RAM and allocate per-pixel count images. Use short converted clips for inspection; streaming and memory-budgeted ingestion are future work.
