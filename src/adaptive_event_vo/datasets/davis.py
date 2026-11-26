"""Convert a bounded DAVIS text-archive clip without unpacking the dataset."""

from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile

import numpy as np

from .calibration import CameraCalibration
from .loader import EventStream
from adaptive_event_vo.provenance import sha256_file

DATASET_PAGE = "https://rpg.ifi.uzh.ch/davis_data.html"
ARCHIVE_URL = "https://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.zip"
DATA_LICENSE = "CC BY-NC-SA 3.0"


def download_shapes_rotation(destination: str | Path) -> Path:
    """Fetch the public archive over verified HTTPS, or reuse a cached file."""
    destination = Path(destination)
    if destination.exists():
        if not zipfile.is_zipfile(destination):
            raise ValueError("cached archive is invalid; remove it and retry")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    try:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise ValueError("downloaded response is not a ZIP archive")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def convert_davis_clip(archive_path: str | Path, output_dir: str | Path, *, start_s: float = 5, duration_s: float = 2) -> dict:
    """Keep [start, start+duration) relative to the first source event.

    Parse decimal timestamps to integer nanoseconds without float epoch loss.
    The exported recording normalizes to its first selected event when loaded;
    its remaining declared duration is saved in the manifest for scheduling.
    The source archive and all exported sensor data remain outside Git.
    """
    if not np.isfinite(start_s) or start_s < 0 or not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("start must be nonnegative and duration positive, both finite")
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    left, duration = Decimal(str(start_s)), Decimal(str(duration_s))
    right = left + duration
    with zipfile.ZipFile(archive_path) as archive:
        def member(suffix):
            names = [name for name in archive.namelist() if name == suffix or name.endswith("/" + suffix)]
            if len(names) != 1:
                raise ValueError(f"archive must contain exactly one {suffix}")
            return names[0]

        images = sorted(name for name in archive.namelist() if name.endswith(".png"))
        if not images:
            raise ValueError("archive has no PNG images to verify sensor dimensions")
        with archive.open(images[0]) as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("invalid PNG header in archive")
        width, height = struct.unpack(">II", header[16:24])
        calibration = CameraCalibration.from_davis_text(archive.read(member("calib.txt")).decode(), width=width, height=height)
        columns = [[], [], [], []]
        origin, previous, reached_end = None, None, False
        with archive.open(member("events.txt")) as handle:
            for number, line in enumerate(io.TextIOWrapper(handle, encoding="utf-8"), start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError(f"invalid events.txt row {number}")
                try:
                    timestamp = Decimal(fields[0])
                except InvalidOperation as exc:
                    raise ValueError(f"invalid timestamp at row {number}") from exc
                if not timestamp.is_finite() or timestamp < 0 or (previous is not None and timestamp < previous):
                    raise ValueError(f"invalid or unordered timestamp at row {number}")
                previous = timestamp
                if origin is None:
                    origin = timestamp
                relative = timestamp - origin
                if relative >= right:
                    reached_end = True
                    break
                if relative < left:
                    continue
                ns = timestamp * 1_000_000_000
                if ns != ns.to_integral_value():
                    raise ValueError("source timestamp precision exceeds integer nanoseconds")
                columns[0].append(int(ns))
                for column, token in zip(columns[1:], fields[1:]):
                    column.append(int(token))
        if not reached_end:
            raise ValueError("source does not cover the requested clip end; shorten the interval")
        if not columns[0]:
            raise ValueError("selected DAVIS interval has no events")
        raw_t = np.asarray(columns[0], dtype=np.int64)
        x, y, p = [np.asarray(column) for column in columns[1:]]
        EventStream((raw_t - raw_t[0]).astype(float) / 1e9, x, y, p, width, height)
        groundtruth = archive.read(member("groundtruth.txt"))
        gt_times = [float(line.split()[0]) for line in groundtruth.decode().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        first_selected_s = Decimal(int(raw_t[0])) / 1_000_000_000
        output_dir.mkdir(parents=True, exist_ok=True)
        events_path = output_dir / "events.npz"
        np.savez_compressed(events_path, t=raw_t, x=x, y=y, p=p)
        metadata = {
            "schema_version": 1, "dataset": "UZH Event-Camera Dataset and Simulator", "sequence": "shapes_rotation",
            "dataset_url": DATASET_PAGE, "archive_url": ARCHIVE_URL,
            "citation": "Mueggler et al., The Event-Camera Dataset and Simulator, IJRR 36(2), 142–149, 2017",
            "license": DATA_LICENSE, "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
            "archive_sha256": sha256_file(archive_path), "events_sha256": sha256_file(events_path),
            "events_file": "events.npz", "width": width, "height": height, "timestamp_unit": "ns", "event_count": len(raw_t),
            "source_timestamp_origin_s": str(origin), "requested_source_relative_start_s": float(left),
            "requested_duration_s": float(duration), "first_selected_source_timestamp_s": str(first_selected_s),
            "normalized_interval_end_s": float(origin + right - first_selected_s),
            "selection_convention": "[source_origin + start, source_origin + start + duration)",
            "preprocessing": "bounded interval only; no sorting, denoising, event dropping or coordinate rectification",
            "calibration": calibration.to_dict(), "calibration_applied": False,
            "calibration_text_sha256": hashlib.sha256(archive.read(member("calib.txt"))).hexdigest(),
            "groundtruth_available": bool(gt_times), "groundtruth_source_timestamp_bounds_s": [min(gt_times), max(gt_times)],
            "groundtruth_sha256": hashlib.sha256(groundtruth).hexdigest(), "groundtruth_used_for_evaluation": False,
        }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata
