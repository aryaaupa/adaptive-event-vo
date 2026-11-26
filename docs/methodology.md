# Methodology and proposed contribution

## Scope and hypothesis

The goal is to test whether a causal adaptive accumulation policy improves the accuracy–computation trade-off of an otherwise fixed monocular event VO pipeline. This document specifies the proposed adaptive/VO experiment, not a validated algorithm or claim of novelty. Weeks 1–2 implement ingestion, exhaustive fixed windows and a validated image-correspondence baseline. See [the implemented Week 2 protocol](week2.md) for the current representation, confidence checks, real-data run and limitations.

There are two separate validation stages: representation/motion quality first, then camera trajectory accuracy. Optical flow on accumulated events is not by itself camera odometry.

## Proposed window controller

After processing window k, estimate event rate and image-motion activity from that completed window and prior history. Choose the next duration using:

```text
r_k = N_k / dt_k                                         [events/s]
r_hat_k = (1 - gamma) * r_hat_(k-1) + gamma * r_k         [events/s]
a_k = median(valid flow speeds) / v_ref                 [dimensionless]
dt_target = N_target / (max(r_hat_k, r_floor) * (1 + beta * a_k))
dt_(k+1) = clip((1 - alpha) * dt_k + alpha * dt_target,
                dt_min, dt_max)                         [s]
```

`N_target` is an event-count budget per representation; `v_ref` is a fixed reference speed in pixels/s. Optical-flow displacements must be divided by the actual interval between frame reference times before calculating speed. `alpha` and `gamma` are smoothing factors in `(0, 1]`; `beta >= 0` controls the motion response. `r_floor` avoids division by zero. Bounds, budgets, reference speed, initialization, and confidence thresholds must be frozen after development-sequence tuning.

The expected behavior is shorter windows in high-rate or high-motion regimes, longer windows in sparse or low-motion regimes, and bounded changes to avoid oscillation. This is a candidate policy; failure to improve results is an admissible outcome.

**Causality:** window k+1 uses only observations available at the end of window k. Start with a fixed warm-up duration. Low-confidence or unavailable flow sets activity to zero and emits a diagnostic flag; zero-event windows remain valid and should move toward the configured maximum duration. Missing tracking must be recorded rather than replaced with fabricated motion. The current code does not implement this policy.

Required controller tests will cover bounds, zero events, sudden rate changes, timestamp duplicates, alternating activity, final-window handling, and invariance of earlier decisions to changes in future events. Density-only, motion-only, and combined versions will be ablated.

## Common downstream pipeline

1. Load calibrated events with a documented time base and sensor geometry.
2. Accumulate identical representations for every policy. Week 2 pools polarity counts for tracking; use that same choice for initial adaptive comparisons. Investigate separate-polarity tracking or time surfaces only as separately controlled experiments.
3. Establish a fixed-window optical-flow/correspondence baseline, with confidence checks and documented frame reference times. Validate on synthetic known pixel displacements before using real recordings.
4. Apply camera intrinsics and distortion correction. Estimate relative pose from geometrically verified correspondences and integrate poses with a declared camera/world convention.
5. Record trajectory timestamps, tracking failures, processed-event counts, representation count, and stage timing.

Monocular two-view geometry generally recovers translation direction rather than metric scale. Pure rotation, low parallax, dynamic objects, accumulated-event artifacts, and sparse correspondences may make pose estimation degenerate. Calibration and observability must be validated before reporting a trajectory. Per-frame scale correction using ground truth would invalidate an unaided monocular VO comparison.

## Evaluation commitments

Use identical sequence intervals, calibration, estimator settings, and hardware for each method. Define one pose timestamp convention and interpolate/synchronize with ground truth only within its supported range. Record unmatched intervals and tracking coverage; never compare only the successful fragments without disclosing failures.

For monocular trajectories, report ATE RMSE after a **single sequence-level Sim(3) alignment** (rotation, translation, one global scale) to ground-truth camera positions at matched timestamps. This compensates for global scale ambiguity, not drift; report the fitted scale. Metric-scale variants, if later introduced, require a separate SE(3)-aligned evaluation. Do not mix alignment protocols in one table.

Report translational and rotational RPE at a predeclared fixed temporal lag, using matched pose pairs and the same global scale where applicable. Do not use a per-frame pair lag when methods output poses at different rates. Specify lag, association tolerance, interpolation, coordinate conventions, units, sample counts, and failure handling in the experiment manifest. Evaluation software must be checked against independently known transformations before use.

Measure processing latency separately from event accumulation delay: a 50 ms window can process quickly yet make events wait up to its duration. State timing boundaries, warm-up, repetitions, thread settings, peak memory, and whether I/O/plotting is excluded. Report p50/p95 per-window processing latency, total sequence compute time, input events/s and processed events/s, plus capture-to-output delay where measurable.

Adaptive windows do **not** inherently reduce the number of input events. Disjoint exhaustive windows ordinarily consume the same events while changing representation count and estimator overhead. Track input events, events actually processed (including repeats for overlapping windows), rejected events, and generated representations separately. Any subsampling or dropping must be declared and applied as an explicit experimental factor.

## Decision gate

An improvement requires evidence across held-out sequences: lower measured compute at comparable trajectory error and tracking coverage, or lower error at comparable compute, relative to tuned fixed policies. Show individual sequences, repeated timing variability, failures, and a Pareto plot. Do not promote a single favorable synthetic visualization into a research result. Assess related work before claiming the controller is novel.
