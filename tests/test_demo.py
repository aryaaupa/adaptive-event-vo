import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_event_vo.cli import main
from adaptive_event_vo.datasets import EventStream, generate_events, load_events, save_events


class DemoTests(unittest.TestCase):
    def test_report_decimal_window_boundaries(self):
        from adaptive_event_vo.visualization.plots import save_report

        events = EventStream(np.array([0, 0.1, 0.2, 0.3, 0.4]), np.zeros(5), np.zeros(5), np.ones(5), 2, 2)
        with tempfile.TemporaryDirectory() as directory:
            summary = save_report(events, Path(directory) / "report.png", duration=0.1)
        self.assertEqual([w["event_count"] for w in summary["windows"]], [1, 1, 1, 1])
        self.assertEqual(summary["windows"][2]["end_s"], 0.3)

    def test_synthetic_reproducibility_and_roundtrip(self):
        original = generate_events(seed=7)
        repeat = generate_events(seed=7)
        changed = generate_events(seed=8)
        for field in ("t", "x", "y", "p"):
            np.testing.assert_array_equal(getattr(original, field), getattr(repeat, field))
        self.assertFalse(np.array_equal(original.x, changed.x))
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".csv", ".npz"):
                path = Path(directory) / f"events{suffix}"
                save_events(original, path)
                loaded = load_events(path, width=128, height=96, timestamp_unit="us")
                np.testing.assert_allclose(loaded.t, original.t, atol=1e-12)
                for field in ("x", "y", "p"):
                    np.testing.assert_array_equal(getattr(loaded, field), getattr(original, field))

    def test_end_to_end_demo_outputs_and_conserved_event_counts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["demo", "--output-dir", directory]), 0)
            root = Path(directory)
            summary = json.loads((root / "event_stream.json").read_text())
            self.assertTrue((root / "event_stream.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            events = load_events(root / "synthetic_events.npz", width=128, height=96, timestamp_unit="us")
            self.assertEqual(summary["event_count"], len(events))
            self.assertEqual(summary["positive_events"] + summary["negative_events"], len(events))
            self.assertEqual(summary["spatial_count_sum"], len(events))
            self.assertEqual(summary["temporal_count_sum"], len(events))
            self.assertEqual(summary["data_kind"], "synthetic")
            self.assertEqual(len(summary["source_sha256"]), 64)
            for window in summary["windows"]:
                self.assertEqual(window["event_count"], len(events.window(window["start_s"], window["end_s"])))

    def test_cli_invalid_input_returns_error_without_figure(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()) as errors:
            output = Path(directory) / "figure.png"
            code = main(["view", str(Path(directory) / "missing.csv"), "--width", "4", "--height", "3", "--timestamp-unit", "us", "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("error:", errors.getvalue())
            self.assertFalse(output.exists())
            code = main(["demo", "--output-dir", directory, "--window-ms", "0"])
            self.assertEqual(code, 2)
            self.assertFalse((Path(directory) / "synthetic_events.npz").exists())


if __name__ == "__main__":
    unittest.main()
