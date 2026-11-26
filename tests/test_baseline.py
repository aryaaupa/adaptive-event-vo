import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from adaptive_event_vo.cli import main
from adaptive_event_vo.datasets import EventStream
from adaptive_event_vo.datasets.motion_fixture import translated_events
from adaptive_event_vo.evaluation.baseline import BaselineConfig, run_baseline

import numpy as np


class BaselineTests(unittest.TestCase):
    def test_end_to_end_known_translation_exports_velocity_and_workload(self):
        events = translated_events(2, -1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_baseline(events, root, config=BaselineConfig(window_ms=(10,), repetitions=2, end_s=0.02))
            method = summary["methods"][0]
            self.assertEqual(method["processed_events_per_repetition"], len(events))
            self.assertEqual(method["processed_events_all_repetitions"], 2 * len(events))
            self.assertEqual(method["representations_per_repetition"], 2)
            self.assertEqual(method["successful_pairs"], 1)
            self.assertEqual(method["pair_status_counts"], {"ok": 1})
            with (root / "windows.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(float(rows[1]["median_vx_px_s"]), 200, delta=10)
            self.assertAlmostEqual(float(rows[1]["median_vy_px_s"]), -100, delta=10)
            with (root / "tracks.csv").open() as handle:
                self.assertGreater(len(list(csv.DictReader(handle))), 20)
            self.assertFalse(summary["camera_pose_estimated"])
            self.assertFalse(summary["trajectory_accuracy_measured"])
            self.assertEqual(json.loads((root / "summary.json").read_text())["methods"][0]["dropped_events"], 0)
            self.assertTrue((root / "baseline.png").read_bytes().startswith(b"\x89PNG"))

    def test_empty_windows_fail_explicitly_without_bridging_gaps(self):
        events = EventStream(np.array([0.001, 0.031]), np.array([5, 5]), np.array([5, 5]), np.array([1, 1]), 24, 24)
        with tempfile.TemporaryDirectory() as directory:
            summary = run_baseline(events, directory, config=BaselineConfig(window_ms=(10, 25, 50), repetitions=1, end_s=0.05))
            self.assertTrue(all(m["processed_events_per_repetition"] == 2 for m in summary["methods"]))
            self.assertEqual(summary["methods"][0]["empty_windows"], 3)
            self.assertEqual(summary["methods"][0]["successful_pairs"], 0)

    def test_cli_validation_and_demo(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            self.assertEqual(main(["validate-flow", "--output-dir", str(root / "validation")]), 0)
            self.assertTrue(json.loads((root / "validation/validation.json").read_text())["passed"])
            self.assertEqual(main(["motion-demo", "--output-dir", str(root / "demo")]), 0)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["baseline", str(root / "demo/synthetic_motion.npz")]), 2)

    def test_invalid_configs(self):
        for values in ({"window_ms": []}, {"window_ms": [10, 10]}, {"window_ms": [-10]}, {"repetitions": 0},
                       {"repetitions": 1.5}, {"end_s": -1}, {"unknown": True}, {"flow": {"unknown": 1}}):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                BaselineConfig.from_dict(values)
