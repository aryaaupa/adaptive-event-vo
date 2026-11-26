import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import numpy as np

from adaptive_event_vo.cli import main
from adaptive_event_vo.datasets import load_events
from adaptive_event_vo.datasets.calibration import CameraCalibration
from adaptive_event_vo.datasets.davis import convert_davis_clip


class DavisTests(unittest.TestCase):
    def archive(self, root, events):
        path = root / "fixture.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("events.txt", events)
            archive.writestr("calib.txt", "20 20 16 12 0 0 0 0 0")
            archive.writestr("groundtruth.txt", "0 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n")
            archive.writestr("images/00000001.png", b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 32, 24))
        return path

    def test_exact_nanoseconds_bounded_selection_metadata_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.archive(root, "0.000000000 0 0 1\n0.100000001 4 5 0\n0.100000002 5 5 1\n0.149999999 6 5 1\n0.150000000 7 5 1\n")
            metadata = convert_davis_clip(archive, root / "clip", start_s=0.1, duration_s=0.05)
            self.assertEqual(metadata["event_count"], 3)
            self.assertEqual(metadata["first_selected_source_timestamp_s"], "0.100000001")
            self.assertEqual(metadata["normalized_interval_end_s"], 0.049999999)
            self.assertFalse(metadata["calibration_applied"])
            self.assertFalse(metadata["groundtruth_used_for_evaluation"])
            events = load_events(root / "clip/events.npz", width=32, height=24, timestamp_unit="ns")
            np.testing.assert_allclose(events.t, [0, 1e-9, 0.049999998], rtol=1e-12, atol=0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["baseline", str(root / "clip/events.npz"), "--manifest", str(root / "clip/manifest.json"), "--output-dir", str(root / "results")]), 0)
            summary = json.loads((root / "results/summary.json").read_text())
            self.assertTrue(all(m["processed_events_per_repetition"] == 3 for m in summary["methods"]))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["baseline", str(root / "clip/events.npz"), "--manifest", str(root / "clip/manifest.json"), "--width", "99"]), 2)
            metadata["events_sha256"] = "incorrect"
            (root / "clip/manifest.json").write_text(json.dumps(metadata))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["baseline", str(root / "clip/events.npz"), "--manifest", str(root / "clip/manifest.json")]), 2)
            (root / "clip/manifest.json").write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["baseline", str(root / "clip/events.npz"), "--manifest", str(root / "clip/manifest.json")]), 2)

    def test_reject_out_of_coverage_unordered_and_bad_pixels(self):
        cases = ("0 0 0 1\n0.01 1 1 1\n", "0 0 0 1\n0.2 1 1 1\n0.1 1 1 1\n0.5 1 1 1\n", "0 0 0 1\n0.1 999 1 1\n0.5 1 1 1\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for events in cases:
                with self.subTest(events=events), self.assertRaises(ValueError):
                    convert_davis_clip(self.archive(root, events), root / "clip", start_s=0.05, duration_s=0.3)

    def test_calibration_contract(self):
        for text in ("1 2", "-1 20 16 12 0 0 0 0 0", "20 20 999 12 0 0 0 0 0", "20 20 16 12 nan 0 0 0 0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                CameraCalibration.from_davis_text(text, width=32, height=24)
