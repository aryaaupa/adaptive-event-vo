"""Validated CSV/NPZ ingestion; timestamps are normalized to relative seconds."""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIME_DIVISORS = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass(frozen=True)
class EventStream:
    """Events in input order, with relative seconds and signed polarity.

    Coordinates use image convention: x increases right, y increases down.
    Equal timestamps are valid. Empty streams represent empty time windows.
    Arrays are copied and made read-only to preserve validation invariants.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    width: int
    height: int
    timestamp_unit: str = "s"
    timestamp_origin: str = "0"

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        arrays = {name: _vector(getattr(self, name), name) for name in ("t", "x", "y", "p")}
        if len({len(a) for a in arrays.values()}) != 1:
            raise ValueError("t, x, y and p must have equal lengths")
        t = arrays["t"]
        if np.any(t < 0) or np.any(t[1:] < t[:-1]):
            raise ValueError("timestamps must be nonnegative and nondecreasing; input is not sorted automatically")
        for name, limit in (("x", self.width), ("y", self.height)):
            a = arrays[name]
            if np.any(a != np.floor(a)) or np.any(a < 0) or np.any(a >= limit):
                raise ValueError(f"{name} must contain integer pixel coordinates in [0, {limit})")
        if not np.all(np.isin(arrays["p"], (-1, 0, 1))):
            raise ValueError("polarity must use 0/1 or -1/+1 values")
        if np.any(arrays["p"] == 0) and np.any(arrays["p"] == -1):
            raise ValueError("polarity mixes 0/1 and -1/+1 encodings")
        arrays["p"] = np.where(arrays["p"] > 0, 1, -1)
        for name, dtype in (("t", np.float64), ("x", np.int64), ("y", np.int64), ("p", np.int8)):
            array = arrays[name].astype(dtype, copy=True)
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    def __len__(self) -> int:
        return len(self.t)

    def window(self, start: float, end: float) -> EventStream:
        """Return [start, end), in seconds relative to the recording origin."""
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError("window bounds must be finite and satisfy 0 <= start < end")
        left, right = np.searchsorted(self.t, [start, end], side="left")
        return EventStream(
            self.t[left:right], self.x[left:right], self.y[left:right], self.p[left:right],
            self.width, self.height, self.timestamp_unit, self.timestamp_origin,
        )


def _relative_timestamps(raw: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    if unit not in TIME_DIVISORS:
        raise ValueError(f"timestamp_unit must be one of {tuple(TIME_DIVISORS)}")
    raw = _vector(raw, "t")
    if not len(raw):
        raise ValueError("event recording is empty")
    if np.any(raw < 0) or np.any(raw[1:] < raw[:-1]):
        raise ValueError("timestamps must be nonnegative and nondecreasing; input is not sorted automatically")
    # Subtract before converting integer epoch timestamps to float. This retains
    # nanosecond differences even when the absolute timestamps exceed 2**53.
    relative = (raw - raw[0]).astype(np.float64) / TIME_DIVISORS[unit]
    if not np.all(np.isfinite(relative)):
        raise ValueError("timestamp range cannot be represented in seconds")
    return relative, str(raw[0])


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[str]] = {name: [] for name in ("t", "x", "y", "p")}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(columns).issubset(reader.fieldnames):
            raise ValueError("CSV header must contain t,x,y,p (column order may vary)")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV contains duplicate column names")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(name) is None or not row[name].strip() for name in columns):
                raise ValueError(f"CSV row {row_number} has missing or extra values")
            for name in columns:
                columns[name].append(row[name].strip())
    result = {}
    for name, values in columns.items():
        try:
            # Integer timestamps preserve large epoch values in nanoseconds.
            if name == "t" and all(v.lstrip("+-").isdigit() for v in values):
                result[name] = np.asarray(values, dtype=np.int64)
            else:
                result[name] = np.asarray(values, dtype=np.float64)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"CSV column {name} contains invalid or out-of-range numbers") from exc
    return result


def load_events(path: str | Path, *, width: int, height: int, timestamp_unit: str) -> EventStream:
    """Load named t,x,y,p arrays from .csv or .npz without guessing units.

    NPZ loading disables pickle. Sensor dimensions are required rather than
    inferred from active pixels. Invalid input is rejected, never repaired.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        arrays = _read_csv(path)
    elif path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                if not {"t", "x", "y", "p"}.issubset(archive.files):
                    raise ValueError("NPZ must contain named arrays t,x,y,p")
                arrays = {name: archive[name] for name in ("t", "x", "y", "p")}
        except (zipfile.BadZipFile, EOFError) as exc:
            raise ValueError("NPZ archive is truncated or corrupt") from exc
    else:
        raise ValueError("supported event formats are .csv and .npz")
    t, origin = _relative_timestamps(arrays["t"], timestamp_unit)
    return EventStream(t, arrays["x"], arrays["y"], arrays["p"], width, height, timestamp_unit, origin)
