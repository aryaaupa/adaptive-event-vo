"""Identical fixed-window tracking pipelines with explicit workload accounting."""

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from adaptive_event_vo.datasets.loader import EventStream
from adaptive_event_vo.datasets.windows import fixed_windows
from adaptive_event_vo.motion import FlowConfig, estimate_flow, tracking_frame
from adaptive_event_vo.provenance import environment


@dataclass(frozen=True)
class BaselineConfig:
    window_ms: tuple[float, ...] = (10, 25, 50)
    repetitions: int = 3
    start_s: float = 0.0
    end_s: float | None = None
    flow: FlowConfig = FlowConfig()

    def __post_init__(self):
        if not self.window_ms or any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in self.window_ms):
            raise ValueError("window_ms must contain finite positive durations")
        if len(set(self.window_ms)) != len(self.window_ms):
            raise ValueError("window_ms values must be unique")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int) or not 1 <= self.repetitions <= 20:
            raise ValueError("repetitions must be an integer in [1, 20]")
        if not np.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("start_s must be finite and nonnegative")
        if self.end_s is not None and (not np.isfinite(self.end_s) or self.end_s <= self.start_s):
            raise ValueError("end_s must be finite and greater than start_s")

    @classmethod
    def from_dict(cls, raw: dict) -> "BaselineConfig":
        data = dict(raw)
        if "flow" in data:
            data["flow"] = FlowConfig(**data["flow"])
        if "window_ms" in data:
            data["window_ms"] = tuple(data["window_ms"])
        return cls(**data)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_baseline(events: EventStream, output_dir: str | Path, *, config: BaselineConfig = BaselineConfig(), provenance: dict | None = None) -> dict:
    """Run every policy on the same interval; no image or trajectory ground truth.

    Timed stages: count/transform/blur and feature detection + forward/backward LK.
    Excluded: loading, schedule construction, warm-up, diagnostics, serialization,
    plotting, and simulated accumulation delay. Every scheduled event is counted.
    """
    if not len(events):
        raise ValueError("baseline requires a nonempty recording")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, tracks, methods = [], [], []
    track_fields = ["window_ms", "window_index", "reference_s", "status", "x", "y", "target_x", "target_y", "vx_px_s", "vy_px_s", "fb_error_px", "lk_error"]
    for duration_ms in config.window_ms:
        timer = perf_counter()
        windows = list(fixed_windows(events, duration_ms / 1000, start_s=config.start_s, end_s=config.end_s))
        schedule_ms = (perf_counter() - timer) * 1000
        if not windows:
            raise ValueError("no windows selected")
        # One untimed pair for each policy, using only that policy's own frames.
        if len(windows) >= 2:
            warm_a, warm_b = [tracking_frame(events, w, config.flow) for w in windows[:2]]
            estimate_flow(warm_a, warm_b, dt_s=windows[1].reference_s - windows[0].reference_s, config=config.flow)
        method_rows, totals, loop_totals = [], [], []
        for repetition in range(config.repetitions):
            previous, previous_window = None, None
            measured_total = 0.0
            loop_start = perf_counter()
            for window in windows:
                tick = perf_counter()
                frame = tracking_frame(events, window, config.flow)
                represented = perf_counter()
                flow = None if previous is None else estimate_flow(previous, frame, dt_s=window.reference_s - previous_window.reference_s, config=config.flow)
                processed = perf_counter()
                representation_ms = (represented - tick) * 1000
                flow_ms = (processed - represented) * 1000 if flow is not None else 0.0
                processing_ms = representation_ms + flow_ms
                measured_total += processing_ms
                ok = flow is not None and flow.status == "ok"
                velocity = np.median(flow.displacement_px, axis=0) / flow.dt_s if ok else None
                row = {
                    "window_ms": duration_ms, "repetition": repetition, "window_index": window.index,
                    "start_s": window.start_s, "end_s": window.end_s, "reference_s": window.reference_s,
                    "source_reference_s": previous_window.reference_s if previous_window else None,
                    "reference_dt_s": flow.dt_s if flow is not None else None,
                    "right_closed": window.right_closed, "partial": window.partial, "event_count": window.event_count,
                    "status": "initialization" if flow is None else flow.status,
                    "detected_tracks": flow.detected if flow is not None else 0,
                    "accepted_tracks": flow.accepted if flow is not None else 0,
                    "median_vx_px_s": float(velocity[0]) if ok else None,
                    "median_vy_px_s": float(velocity[1]) if ok else None,
                    "median_speed_px_s": float(np.median(np.linalg.norm(flow.displacement_px, axis=1)) / flow.dt_s) if ok else None,
                    "median_fb_error_px": float(np.median(flow.fb_error_px[flow.valid])) if flow is not None and flow.accepted else None,
                    "representation_ms": representation_ms, "flow_ms": flow_ms, "processing_ms": processing_ms,
                }
                method_rows.append(row)
                if repetition == 0 and flow is not None:
                    for i in np.flatnonzero(flow.valid):
                        displacement = (flow.target_xy[i] - flow.source_xy[i]) / flow.dt_s
                        tracks.append(dict(zip(track_fields, [duration_ms, window.index, window.reference_s, flow.status,
                            *map(float, flow.source_xy[i]), *map(float, flow.target_xy[i]), *map(float, displacement), float(flow.fb_error_px[i]), float(flow.lk_error[i])])))
                previous, previous_window = frame, window
            loop_totals.append((perf_counter() - loop_start) * 1000)
            totals.append(measured_total)
        first = [r for r in method_rows if r["repetition"] == 0]
        counts = int(sum(w.event_count for w in windows))
        begin = int(np.searchsorted(events.t, config.start_s, side="left"))
        stop = int(np.searchsorted(events.t, windows[-1].end_s, side="right" if windows[-1].right_closed else "left"))
        if counts != stop - begin:
            raise RuntimeError("window scheduler did not conserve selected events")
        pairs = first[1:]
        methods.append({
            "window_ms": duration_ms, "interval_start_s": windows[0].start_s, "interval_end_s": windows[-1].end_s,
            "final_window_right_closed": windows[-1].right_closed, "final_window_partial": windows[-1].partial,
            "representations_per_repetition": len(windows), "flow_pairs_per_repetition": len(pairs),
            "processed_events_per_repetition": counts, "processed_events_all_repetitions": counts * config.repetitions,
            "dropped_events": 0, "empty_windows": sum(w.event_count == 0 for w in windows),
            "successful_pairs": sum(r["status"] == "ok" for r in pairs),
            "successful_pair_fraction": sum(r["status"] == "ok" for r in pairs) / len(pairs) if pairs else None,
            "pair_status_counts": dict(Counter(r["status"] for r in pairs)),
            "median_accepted_tracks_per_pair": float(np.median([r["accepted_tracks"] for r in pairs])) if pairs else None,
            "processing_ms_p50": float(np.percentile([r["processing_ms"] for r in method_rows], 50)),
            "processing_ms_p95": float(np.percentile([r["processing_ms"] for r in method_rows], 95)),
            "total_processing_ms_per_repetition": totals, "total_processing_ms_median": float(np.median(totals)),
            "wall_loop_ms_per_repetition": loop_totals, "schedule_construction_ms": schedule_ms,
        })
        rows.extend(method_rows)
    if len({m["processed_events_per_repetition"] for m in methods}) != 1:
        raise RuntimeError("policies selected different event workloads")
    summary = {
        "schema_version": 1, "experiment": "fixed_window_image_motion", "camera_pose_estimated": False,
        "trajectory_accuracy_measured": False, "config": asdict(config), "environment": environment(),
        "recording_event_count": len(events), "sensor_width": events.width, "sensor_height": events.height,
        "timestamp_origin": events.timestamp_origin, "input_timestamp_unit": events.timestamp_unit,
        "frame_reference": "window midpoint, including final partial window", "calibration_applied": False,
        "warmup_pairs_per_method": 1 if min(m["representations_per_repetition"] for m in methods) >= 2 else 0,
        "timing_scope": "count+log+blur and feature+forward/backward LK; excludes loading, scheduling, warmup, diagnostics, export, plotting and accumulation delay",
        "timing_interpretation": "exploratory local processing measurements, not a real-time or energy benchmark",
        "status_and_track_statistics": "first repetition; timing percentiles pool all repetitions",
        "provenance": provenance or {}, "methods": methods,
    }
    write_csv(output_dir / "windows.csv", rows)
    write_csv(output_dir / "tracks.csv", tracks, track_fields)
    from adaptive_event_vo.visualization.motion import plot_baseline
    plot_baseline(rows, summary, output_dir / "baseline.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
