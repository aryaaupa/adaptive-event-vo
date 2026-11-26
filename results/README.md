# Recorded artifacts

These files are outputs from actual local runs, not illustrative performance claims.

| Artifact | Scope |
| --- | --- |
| `ingestion_demo.json`, `figures/event_stream.png` | Week 1 seeded synthetic ingestion diagnostics, package 0.1.0 |
| `tables/flow_validation.json`, `figures/flow_validation.png` | Week 2 known count-image displacement and explicit failure cases |
| `tables/shapes_rotation_baseline.json`, `tables/shapes_rotation_methods.csv`, `figures/fixed_window_baseline.png` | Week 2 image-motion baseline on real DAVIS `shapes_rotation`, source interval [5, 7) seconds |

The Week 2 summary records input/archive hashes, full configuration, code fingerprint, dependency versions, CPU/thread settings, excluded timing stages and tracking failures. The compact CSV is derived from its per-method fields; the full `windows.csv` and `tracks.csv` are regenerated under `outputs/`.

No artifact contains an estimated camera trajectory, ATE/RPE, measured power consumption, or a demonstrated adaptive-method improvement. The different package versions on Week 1 and Week 2 artifacts preserve their original provenance.

See [the Week 2 protocol](../docs/week2.md) to reproduce the new runs. Synthetic fixtures/figures are project-generated under MIT. DAVIS-derived result artifacts carry attribution to Mueggler et al., *The Event-Camera Dataset and Simulator*, IJRR 2017, [dataset source](https://rpg.ifi.uzh.ch/davis_data.html); [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). No dataset archive, raw event clip or intensity image is included in Git.
