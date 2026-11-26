# Week 2: fixed-window image-motion baseline

The working pipeline is **events → exhaustive fixed windows → count images → sparse correspondences → image velocity and workload reports**. It uses events rather than the dataset's ordinary intensity frames. It does not estimate camera poses, integrate a trajectory, or measure ATE/RPE.

## Reproduce without downloading data

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
adaptive-event-vo validate-flow --output-dir outputs/flow_validation
adaptive-event-vo motion-demo --output-dir outputs/motion_demo
```

`validate-flow` creates four independently specified count-image translations: `(2, -1)`, `(-3, 2)`, `(0, 0)`, and `(1, 3)` pixels. Endpoint error compares every accepted vector with that known displacement. Acceptance requires at least 20 tracks, mean error below 0.2 pixels, and p95 error below 0.5 pixels. Blank, textureless, and disappearing-pattern cases must produce explicit failures. The unit suite also checks a fractional `(1.5, -0.75)` pixel warp.

The synthetic stream carries exactly repeated texture in two 10 ms windows, with no occlusion or sensor artifacts. Very small errors on it validate sign, units, indexing, image construction, and the LK implementation; they do not establish event-camera accuracy under realistic appearance changes. `motion-demo` exercises file saving, loading, scheduling, tracking, export, and plotting on that fixture.

## Reproduce the real-event run

The selected **development clip** is `shapes_rotation`, `[5, 7)` seconds relative to the original first event. It was selected to exercise tracking during motion, not as a held-out accuracy benchmark. Data and calibration come from the [UZH Event-Camera Dataset and Simulator](https://rpg.ifi.uzh.ch/davis_data.html), Mueggler et al., IJRR 2017. Its [CC BY-NC-SA 3.0 terms](https://creativecommons.org/licenses/by-nc-sa/3.0/) apply to sensor data; the code license does not replace them.

```bash
# Downloads approximately 157 MB once, or reuses the cached archive.
python scripts/download_data.py --start-s 5 --duration-s 2

adaptive-event-vo baseline data/davis/shapes_rotation_clip/events.npz \
  --manifest data/davis/shapes_rotation_clip/manifest.json \
  --config configs/fixed_window.json \
  --output-dir outputs/week2/shapes_rotation
```

The downloader supports this named sequence only. It reads archive members directly rather than extracting files, converts selected decimal timestamps to integer nanoseconds, validates the selected events, and saves a small NPZ plus manifest. It stops after reaching the requested clip end. Sensor dimensions are verified from an archived image header; intensity images are not fed into tracking. The manifest includes the source URL, archive/event hashes, calibration, ground-truth coverage/hash, and conversion interval.

For the reference archive, SHA-256 is `56aade6bf53dcf73e8fe40905ccac8385cd7606bc9a85103bf2c9f9045117551`. This is our observed downloaded-file hash, not an upstream signature. A different archive hash must be investigated before comparing reproduction results.

The reference clip contains **437,670 events on a 240×180 sensor**. The first selected source timestamp is `5.000001001` seconds. Loading places that event at zero; the declared normalized exclusive endpoint is `1.999998999` seconds, so the original source boundary remains exactly 7 seconds. Manifest/input hash mismatches and conflicting explicit dimensions/units are rejected.

For your own CSV/NPZ, supply `--width`, `--height`, and `--timestamp-unit` instead of a DAVIS manifest. Configuration is strict JSON; unknown keys, invalid durations and invalid flow settings are errors. Use the same configuration for comparisons.

## Scheduling and frame timestamps

`datasets.windows.fixed_windows` partitions the whole requested interval, including silent windows, using decimal arithmetic for duration boundaries and binary searches over normalized timestamps. Empty windows are processed and reported, not skipped. No event is duplicated by overlapping windows.

With an explicit endpoint, **all windows are half-open** and an event exactly at the endpoint is excluded. With no endpoint, the final event timestamp is treated as the endpoint and **only the final window is right-closed** so the final event is retained exactly once. The terminal exception applies even when the endpoint equals a nominal window boundary. A singleton recording uses one nominal window. These conventions differ from the Week 1 viewer, which deliberately shows only four half-open diagnostic intervals.

Final partial windows are included. Each frame's declared reference timestamp is its interval midpoint; flow velocity is displacement divided by the actual difference between the two midpoints. This includes the shorter final interval. These reference times describe accumulated representations, not instantaneous exposures or camera-pose timestamps.

## Representation and correspondence estimator

1. Count every event at its original distorted pixel coordinate. Pool polarities without cancellation for this baseline; Week 1's separate-polarity count API remains available.
2. Apply the same fixed transform in every frame: `round(255 * log(1 + min(count, 4)) / log(5))`, then Gaussian blur with sigma 0.8 pixels. Clipping limits representation intensity, not the processed-event counter. There is no per-frame maximum normalization.
3. Detect Shi–Tomasi corners anew in the source frame of each pair. Defaults: at most 300 corners, quality 0.01, separation 5 px, block size 7. This is pairwise correspondence estimation, not persistent feature tracks.
4. Run OpenCV pyramidal Lucas–Kanade forward and backward: 21×21 patch, maximum pyramid level 3, 30 iterations, epsilon 0.01, minimum eigenvalue 0.0001.
5. Require forward/backward LK success, finite in-image endpoints with a border margin, round-trip error ≤1 px, and LK photometric error ≤30 in both directions.

At least eight accepted correspondences make a pair pass the tracking gate. Photometric and round-trip errors are confidence checks, not ground-truth accuracy. A pair that fails the gate exports its reason and blank median velocity fields. Valid individual correspondences below the gate can still appear in `tracks.csv`, with the failed pair status attached. An empty or bad pair never bridges to a later frame. All thresholds are stored in [the configuration](../configs/fixed_window.json).

Calibration is parsed, checked and attached to the result, but **distortion correction and geometric pose estimation are not applied** in Week 2. Motion is measured in distorted sensor pixels. Ground truth is available in the archive but is not used to calculate image flow error. Rotation-only data also cannot validate recovery of monocular translational scale.

## Export and timing contract

- `summary.json`: complete configuration, implementation SHA-256, runtime and CPU metadata, dataset provenance, per-method workload/failure counts and timing repetitions.
- `windows.csv`: every window in every repetition, bounds, midpoint/delta, event count, status, accepted/detected tracks, median velocity, and representation/flow processing times.
- `tracks.csv`: accepted correspondence coordinates, velocity, and confidence errors from the first repetition.
- `baseline.png`: event counts, image speed with gaps for failures, accepted tracks and local processing-time percentiles.

OpenCV uses one thread and OpenCL is disabled. Each method warms up on its first pair once, then runs three repetitions. Count construction, log transform, blur, corner detection and forward/backward LK are timed with `perf_counter`. Input loading, schedule construction, warm-up, diagnostic aggregation, exports and plotting are excluded; scheduling time and a separate wall-loop time are also recorded. Median total processing time is the median of summed timed stages across repetitions. Percentiles pool windows from all repetitions, including the first representation-only window.

The code does not simulate waiting for events. **Accumulation delay is excluded from processing time**, and energy, peak memory, pose accuracy and real-time deadlines are unmeasured. The single-clip timing is an exploratory local measurement on a shared host, not the Week 5 benchmark.

## What the evidence supports

All three schedules consume the same 437,670 events per repetition while generating 200, 80 and 40 representations. They pass the eight-track gate on 196/199, 76/79 and 37/39 pairs respectively. Lower representation count changes processing work; it does not reduce input events. Pair pass rates at different temporal spacings are descriptive counts, not directly comparable accuracy scores or trajectory tracking coverage.

The [saved real-run summary](../results/tables/shapes_rotation_baseline.json), [method table](../results/tables/shapes_rotation_methods.csv), and [synthetic validation](../results/tables/flow_validation.json) are actual outputs. Per-window and per-track files remain in ignored `outputs/`; rerun the commands to regenerate them. Raw sensor events are not committed.

The next milestone is the causal adaptive controller and its ablations. The remaining research gate is held-out, geometrically validated VO evaluation under a shared accuracy and timing protocol.
