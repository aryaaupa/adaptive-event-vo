"""Image-motion evidence with visible missing estimates and workload counts."""

from pathlib import Path

import numpy as np


def plot_baseline(rows: list[dict], summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for method in summary["methods"]:
        selected = [r for r in rows if r["repetition"] == 0 and r["window_ms"] == method["window_ms"]]
        times = [r["reference_s"] for r in selected]
        label = f"{method['window_ms']:g} ms"
        axes[0, 0].plot(times, [r["event_count"] for r in selected], label=label, linewidth=1)
        axes[0, 1].plot(times, [r["median_speed_px_s"] if r["median_speed_px_s"] is not None else np.nan for r in selected], label=label, linewidth=1)
        axes[1, 0].plot(times[1:], [r["accepted_tracks"] for r in selected[1:]], label=label, linewidth=1)
    axes[0, 0].set(title="Accumulated events · all events consumed", ylabel="Events / window")
    axes[0, 1].set(title="Median image speed · gaps mark tracking failure", ylabel="Pixels / second")
    axes[1, 0].set(title="Accepted correspondences · first repetition", ylabel="Tracks / frame pair")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("Seconds from first selected event")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    methods = summary["methods"]
    x = np.arange(len(methods))
    axes[1, 1].bar(x - 0.18, [m["processing_ms_p50"] for m in methods], width=0.36, label="p50")
    axes[1, 1].bar(x + 0.18, [m["processing_ms_p95"] for m in methods], width=0.36, label="p95")
    axes[1, 1].set(xticks=x, xticklabels=[f"{m['window_ms']:g} ms" for m in methods], title="Local processing time · accumulation delay excluded", ylabel="Milliseconds / window", xlabel="Fixed accumulation duration")
    axes[1, 1].legend(frameon=False)
    dataset = summary.get("provenance", {}).get("dataset", {}).get("sequence", "supplied events")
    fig.suptitle(f"Fixed-window baseline · {dataset}\nImage-plane correspondences only; no camera trajectory or real-data accuracy measurement", fontweight="bold", fontsize=13)
    try:
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
