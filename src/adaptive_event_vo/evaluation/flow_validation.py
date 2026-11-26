"""Quantitative image-displacement tests with an independently known answer."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from adaptive_event_vo.datasets.motion_fixture import translated_events
from adaptive_event_vo.datasets.windows import fixed_windows
from adaptive_event_vo.motion import FlowConfig, estimate_flow, tracking_frame
from adaptive_event_vo.provenance import environment


def validate_flow(output_dir: str | Path, config: FlowConfig = FlowConfig()) -> dict:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, visual = [], None
    for dx, dy in ((2, -1), (-3, 2), (0, 0), (1, 3)):
        events = translated_events(dx, dy)
        windows = list(fixed_windows(events, 0.01, end_s=0.02))
        a, b = [tracking_frame(events, w, config) for w in windows]
        flow = estimate_flow(a, b, dt_s=windows[1].reference_s - windows[0].reference_s, config=config)
        errors = np.linalg.norm(flow.displacement_px - [dx, dy], axis=1)
        mean, p95 = (float(np.mean(errors)), float(np.percentile(errors, 95))) if len(errors) else (None, None)
        passed = flow.status == "ok" and flow.accepted >= 20 and mean < 0.2 and p95 < 0.5
        cases.append({"displacement_px": [dx, dy], "status": flow.status, "accepted_tracks": flow.accepted,
                      "detected_tracks": flow.detected, "mean_endpoint_error_px": mean, "p95_endpoint_error_px": p95, "passed": bool(passed)})
        if visual is None:
            visual = (a, flow)
    failure_cases = []
    a = visual[0]
    for name, source, target, expected in (
        ("blank", np.zeros_like(a), np.zeros_like(a), "empty_source"),
        ("disappearing_texture", a, np.zeros_like(a), "empty_target"),
        ("textureless", np.full_like(a, 128), np.full_like(a, 128), "no_features"),
    ):
        result = estimate_flow(source, target, dt_s=0.01, config=config)
        failure_cases.append({"case": name, "status": result.status, "accepted_tracks": result.accepted,
                              "passed": result.status == expected and result.accepted == 0})
    summary = {"schema_version": 1, "experiment": "known_count_image_translation", "seed": 19,
               "physical_event_camera_simulation": False, "config": asdict(config), "environment": environment(),
               "acceptance": {"min_accepted_tracks": 20, "max_mean_endpoint_error_px": 0.2, "max_p95_endpoint_error_px": 0.5},
               "cases": cases, "failure_cases": failure_cases,
               "passed": all(c["passed"] for c in cases + failure_cases)}
    _plot(cases, visual, output_dir / "validation.png")
    (output_dir / "validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _plot(cases, visual, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.80, wspace=0.35)
    image, flow = visual
    points, vectors = flow.source_xy[flow.valid], flow.displacement_px
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].quiver(points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1], angles="xy", scale_units="xy", scale=0.25, color="#e5a520", width=0.004)
    axes[0].set(title="Known (+2, −1) px translation · arrows ×4", xlabel="x (pixel)", ylabel="y (pixel)")
    errors = [c["mean_endpoint_error_px"] or 0 for c in cases]
    axes[1].bar(np.arange(len(cases)), errors, color="#2466ab")
    axes[1].set_yscale("symlog", linthresh=1e-5)
    for index, error in enumerate(errors):
        axes[1].annotate(f"{error:.2g}", (index, error), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    axes[1].axhline(0.2, color="#c9532d", linestyle="--", label="Mean error acceptance threshold")
    axes[1].set(xticks=np.arange(len(cases)), xticklabels=[str(c["displacement_px"]) for c in cases], xlabel="Known displacement (dx, dy) pixels", ylabel="Mean endpoint error (pixels; symlog)", title="Accepted correspondences · synthetic texture")
    axes[1].set_ylim(0, 0.6)
    axes[1].legend(frameon=True, fontsize=7, loc="upper right")
    fig.suptitle("Controlled image-motion validation · no camera trajectory", fontweight="bold", y=0.98)
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
