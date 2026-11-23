import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_event_vo.datasets import EventStream, load_events


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def csv(self, text):
        path = self.root / "events.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def load(self, path, unit="us"):
        return load_events(path, width=4, height=3, timestamp_unit=unit)

    def test_named_columns_normalization_and_equal_timestamps(self):
        events = self.load(self.csv("x,p,t,y\n1,0,1000,2\n2,1,1000,1\n3,1,2500,0\n"))
        np.testing.assert_allclose(events.t, [0, 0, 0.0015])
        np.testing.assert_array_equal(events.p, [-1, 1, 1])
        self.assertEqual(events.timestamp_origin, "1000")
        with self.assertRaises(ValueError):
            events.t[0] = 1

    def test_integer_nanosecond_epoch_precision_csv_and_npz(self):
        epoch = 1_700_000_000_000_000_000
        csv = self.csv(f"t,x,y,p\n{epoch},0,0,1\n{epoch + 1},1,1,-1\n{epoch + 17},2,2,1\n")
        archive = self.root / "events.npz"
        np.savez(archive, t=np.array([epoch, epoch + 1, epoch + 17]), x=[0, 1, 2], y=[0, 1, 2], p=[1, -1, 1])
        for path in (csv, archive):
            with self.subTest(format=path.suffix):
                np.testing.assert_allclose(self.load(path, "ns").t, [0, 1e-9, 17e-9], rtol=1e-12, atol=0)

    def test_units(self):
        path = self.csv("t,x,y,p\n0,0,0,1\n1,1,1,-1\n")
        for unit, seconds in (("s", 1), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
            with self.subTest(unit=unit):
                self.assertEqual(self.load(path, unit).t[-1], seconds)
        with self.assertRaisesRegex(ValueError, "timestamp_unit"):
            self.load(path, "unknown")

    def test_invalid_recordings_are_rejected(self):
        cases = {
            "unordered": "t,x,y,p\n2,0,0,1\n1,1,1,0\n",
            "negative_time": "t,x,y,p\n-1,0,0,1\n",
            "fractional_coordinate": "t,x,y,p\n0,0.5,0,1\n",
            "out_of_bounds": "t,x,y,p\n0,4,0,1\n",
            "negative_coordinate": "t,x,y,p\n0,-1,0,1\n",
            "invalid_polarity": "t,x,y,p\n0,0,0,2\n",
            "mixed_polarity": "t,x,y,p\n0,0,0,0\n1,0,0,-1\n",
            "nan": "t,x,y,p\nnan,0,0,1\n",
            "infinity": "t,x,y,p\n0,inf,0,1\n",
            "missing_header": "0,0,0,1\n",
            "duplicate_header": "t,x,y,p,p\n0,0,0,1,1\n",
            "missing_value": "t,x,y,p\n0,0,,1\n",
            "extra_value": "t,x,y,p\n0,0,0,1,2\n",
            "nonnumeric": "t,x,y,p\nabc,0,0,1\n",
            "empty": "t,x,y,p\n",
        }
        for name, text in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.load(self.csv(text))

    def test_npz_schema_and_shapes(self):
        path = self.root / "invalid.npz"
        for arrays in (
            {"t": [0], "x": [0], "y": [0]},
            {"t": [0, 1], "x": [0], "y": [0], "p": [1]},
            {"t": [[0]], "x": [0], "y": [0], "p": [1]},
            {"t": np.array([object()]), "x": [0], "y": [0], "p": [1]},
        ):
            with self.subTest(arrays=tuple(arrays)), self.assertRaises(ValueError):
                np.savez(path, **arrays)
                self.load(path)

    def test_sensor_dimensions_are_not_guessed(self):
        for width, height in ((0, 3), (4, -1), (4.5, 3), (True, 3)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                EventStream(np.array([0]), np.array([0]), np.array([0]), np.array([1]), width, height)

    def test_decimal_second_boundaries_after_unit_conversion(self):
        events = self.load(self.csv("t,x,y,p\n0,0,0,1\n100000,1,1,1\n200000,1,1,-1\n300000,1,1,1\n400000,1,1,-1\n"))
        np.testing.assert_array_equal(events.t, [0, 0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(events.window(0.2, 0.3)), 1)
        self.assertEqual(len(events.window(0.3, 0.4)), 1)

    def test_half_open_windows_preserve_origin_and_do_not_double_count(self):
        events = self.load(self.csv("t,x,y,p\n0,0,0,1\n1000,1,1,-1\n1000,1,1,1\n2000,2,2,1\n"))
        first, second = events.window(0, 0.001), events.window(0.001, 0.002)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        np.testing.assert_allclose(second.t, [0.001, 0.001])
        self.assertEqual(len(events.window(0.003, 0.004)), 0)
        self.assertEqual(len(events.window(0.002, 0.003)), 1)
        for bounds in ((0, 0), (-1, 1), (0, float("nan")), (1, 0)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                events.window(*bounds)


if __name__ == "__main__":
    unittest.main()
