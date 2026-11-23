"""Save a diagnostic report without a display server."""

from decimal import Decimal
from pathlib import Path

import numpy as np

from adaptive_event_vo.datasets.loader import EventStream
from adaptive_event_vo.representations.event_frame import event_counts, polarity_image


def save_report(
    events: EventStream, output: str | Path, *, start: float = 0.0,
    duration: float = 0.1, title: str = "Event-stream diagnostics",
) -> dict:
    """Render all-stream activity and four adjacent [start, end) windows.

    The raster is deterministically subsampled to at most 5,000 points. Counts,
    rate and accumulation frames use every event. Final histogram bin includes
    the final timestamp; half-open accumulation windows keep their stated bounds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not len(events):
        raise ValueError("cannot plot an empty recording")
    if not np.isfinite(start) or start < 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("start must be nonnegative and duration must be positive, both finite")
    output = Path(output)
    if output.suffix.lower() not in (".png", ".svg", ".pdf"):
        raise ValueError("figure output must end in .png, .svg or .pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Decimal arithmetic on the stated second values avoids 3 * 0.1 becoming
    # 0.30000000000000004 and assigning an event at 300 ms to the earlier window.
    bounds = [float(Decimal(str(start)) + i * Decimal(str(duration))) for i in range(5)]
    fig = plt.figure(figsize=(13, 9), layout="constrained", facecolor="white")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 1])
    colors = ["#f2661a", "#1466d9"]
    raster = fig.add_subplot(grid[0, :2])
    sample = np.linspace(0, len(events) - 1, min(5000, len(events)), dtype=int)
    for polarity, color, label in ((-1, colors[0], "Negative"), (1, colors[1], "Positive")):
        selected = sample[events.p[sample] == polarity]
        raster.scatter(events.t[selected], events.x[selected], s=3, color=color, alpha=0.5, label=label, rasterized=True)
    raster.set(xlabel="Time from first event (s)", ylabel="x (pixel)", title=f"Asynchronous events · {len(sample):,} displayed")
    raster.legend(frameon=False, markerscale=3, fontsize=8)
    raster.grid(alpha=0.15)

    accumulation = fig.add_subplot(grid[0, 2:])
    accumulation.imshow(polarity_image(events), interpolation="nearest")
    accumulation.set(title="Polarity occupancy · full recording", xlabel="x (pixel)", ylabel="y (pixel)")

    density = fig.add_subplot(grid[1, :2])
    total_counts = event_counts(events).sum(axis=0)
    heatmap = density.imshow(np.log1p(total_counts), cmap="magma", interpolation="nearest")
    density.set(title="Spatial event density · full recording", xlabel="x (pixel)", ylabel="y (pixel)")
    fig.colorbar(heatmap, ax=density, label="log(1 + events / pixel)", shrink=0.85)

    rate = fig.add_subplot(grid[1, 2:])
    span = max(float(events.t[-1]), 1e-6)
    edges = np.linspace(0, span, 41)
    histogram, _ = np.histogram(events.t, bins=edges)
    rate.stairs(histogram / np.diff(edges) / 1000, edges, color="#203b55", fill=True, alpha=0.8)
    rate.axvspan(bounds[0], bounds[-1], color="#f6b745", alpha=0.25, label="Four windows below")
    rate.set(title="Temporal event density · 40 equal bins", xlabel="Time from first event (s)", ylabel="Event rate (k events/s)")
    rate.grid(alpha=0.15)
    rate.legend(frameon=False, fontsize=8)

    windows = []
    for index in range(4):
        left, right = bounds[index:index + 2]
        window = events.window(left, right)
        axis = fig.add_subplot(grid[2, index])
        axis.imshow(polarity_image(window), interpolation="nearest")
        axis.set(title=f"[{left * 1000:g}, {right * 1000:g}) ms\n{len(window):,} events", xlabel="x (pixel)")
        if index == 0:
            axis.set_ylabel("y (pixel)")
        windows.append({"start_s": left, "end_s": right, "event_count": len(window)})
    legend = [Patch(color=color, label=label) for color, label in ((colors[1], "Positive"), (colors[0], "Negative"), ("#853ba6", "Both polarities"))]
    fig.legend(handles=legend, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle(f"{title}\n{len(events):,} events · {events.width} × {events.height} pixels · {events.t[-1]:.3f} s observed span", fontsize=15, fontweight="bold")
    try:
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    return {
        "event_count": len(events), "sensor_width": events.width, "sensor_height": events.height,
        "observed_span_s": float(events.t[-1] - events.t[0]),
        "positive_events": int(np.count_nonzero(events.p > 0)),
        "negative_events": int(np.count_nonzero(events.p < 0)),
        "timestamp_origin": events.timestamp_origin, "input_timestamp_unit": events.timestamp_unit,
        "internal_timestamp_unit": "s", "raster_displayed_events": len(sample),
        "spatial_count_sum": int(total_counts.sum()), "temporal_count_sum": int(histogram.sum()),
        "window_convention": "[start_s, end_s)", "windows": windows,
    }
