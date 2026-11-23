# Six-week implementation tracker

**4 / 24 tasks complete (17% rounded).** Week 1 is implemented locally. Weeks are milestones rather than guaranteed deadlines; VO may require additional time.

## Week 1 — Dataset ingestion and event visualization · 4/4

- [x] Create an installable Python package, CLI, environment instructions and CI configuration.
- [x] Implement CSV/NPZ loading, explicit timestamp units and validation.
- [x] Visualize polarity, spatial/temporal event density and accumulation windows.
- [x] Add a small synthetic recording and automated loader, representation and demo tests.

**Deliverable:** `adaptive-event-vo demo` produces a headless event-viewer report and provenance JSON; dataset documentation and a committed example accompany it. A real sensor recording and interactive GUI are not part of this deliverable.

## Week 2 — Fixed-window motion baseline · 0/4

- [ ] Implement complete 10/25/50 ms scheduling with final-window and boundary tests.
- [ ] Add optical-flow/correspondence estimation with confidence masks and declared reference timestamps.
- [ ] Validate known image displacement and failure cases on controlled synthetic pairs.
- [ ] Run documented real-event clips with calibration metadata and save a reproducible motion baseline.

**Deliverable:** one command that runs a fixed-window motion baseline; flow validation is kept distinct from VO validation.

## Week 3 — Adaptive event processing · 0/4

- [ ] Implement causal event-rate and past-motion activity estimation.
- [ ] Implement the bounded, smoothed controller with startup and missing-flow fallbacks.
- [ ] Test causality, sparse windows, rate changes, bounds and stability.
- [ ] Compare fixed-count, density-only, motion-only and combined schedules with actual workload counters.

**Deliverable:** a controller study with frozen configurations, causal logs, and no trajectory claims before Week 4 validation.

## Week 4 — Visual odometry pipeline · 0/4

- [ ] Validate calibration, distortion correction, time synchronization and coordinate conventions.
- [ ] Estimate relative camera poses from verified correspondences with degeneracy handling.
- [ ] Integrate timestamped trajectories and record tracking loss and reinitialization.
- [ ] Validate ATE/RPE, global scale alignment and ground-truth association against known transformations.

**Deliverable:** a traceable estimated-versus-ground-truth trajectory on a documented real sequence, with scale/coverage limitations disclosed.

## Week 5 — Experiments and benchmarking · 0/4

- [ ] Freeze development/test splits, configurations and the common evaluation protocol.
- [ ] Run all five baseline/adaptive policies on the same held-out intervals.
- [ ] Measure latency, delay, total compute, event workload, memory and tracking coverage with repeated timing.
- [ ] Publish per-sequence results, ablations, failures and an accuracy–computation trade-off figure.

**Deliverable:** reproducible evidence that supports or rejects the controller hypothesis.

## Week 6 — Research presentation and release · 0/4

- [ ] Review related adaptive event-processing work and state the precise contribution.
- [ ] Update the README with measured figures, honest conclusions and limitations.
- [ ] Document full reproduction from data conversion to evaluation artifacts.
- [ ] Run a clean-install check, review public artifacts and prepare a versioned release.

**Deliverable:** a small inspectable research repository with substantiated claims and a credible next research question.
