"""A deterministic moving-edge fixture, not a physical event-camera simulator."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .loader import EventStream


def generate_events(*, seed: int = 7, width: int = 128, height: int = 96) -> EventStream:
    """Generate one second of moving edges, changing activity and background noise.

    No poses or intensity-threshold sensor model are provided; this fixture
    tests ingestion and plotting, not odometry accuracy or physical realism.
    """
    if width < 16 or height < 16:
        raise ValueError("synthetic sensor dimensions must each be at least 16")
    rng = np.random.default_rng(seed)
    all_t, all_x, all_y, all_p = [], [], [], []
    for tick in range(200):
        time = tick / 200
        activity = 0.5 + 0.5 * np.sin(2 * np.pi * time - np.pi / 2)
        count = int(24 + 120 * activity)
        edge = width * (0.15 + 0.65 * time)
        y = rng.integers(height // 6, 5 * height // 6, size=count)
        p = rng.choice([-1, 1], size=count)
        x = np.clip(np.rint(edge + np.where(p > 0, 0, width * 0.12) + rng.normal(0, 0.65, count)), 0, width - 1)
        all_t.append(time + rng.uniform(0, 0.005, count))
        all_x.append(x)
        all_y.append(y)
        all_p.append(p)
    noise_count = 1000
    all_t.append(rng.uniform(0, 1, noise_count))
    all_x.append(rng.integers(0, width, noise_count))
    all_y.append(rng.integers(0, height, noise_count))
    all_p.append(rng.choice([-1, 1], noise_count))
    t = np.rint(np.concatenate(all_t) * 1e6).astype(np.int64)
    order = np.argsort(t, kind="stable")
    t = (t[order] - t.min()).astype(np.float64) / 1_000_000
    return EventStream(t, np.concatenate(all_x)[order], np.concatenate(all_y)[order], np.concatenate(all_p)[order], width, height)


def save_events(events: EventStream, path: str | Path) -> None:
    """Write canonical events using integer microseconds, with <=0.5 us rounding."""
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".npz"):
        raise ValueError("output must be .csv or .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    t_us = np.rint(events.t * 1e6).astype(np.int64)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, t=t_us, x=events.x, y=events.y, p=events.p)
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "x", "y", "p"])
            writer.writerows(zip(t_us, events.x, events.y, events.p))
