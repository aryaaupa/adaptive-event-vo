import unittest

import numpy as np

from adaptive_event_vo.datasets import EventStream
from adaptive_event_vo.datasets.windows import fixed_windows


def stream(t):
    return EventStream(np.asarray(t), np.zeros(len(t)), np.zeros(len(t)), np.ones(len(t)), 4, 3)


class WindowTests(unittest.TestCase):
    def test_complete_schedule_preserves_duplicates_silence_and_final_event(self):
        events = stream([0, 0.01, 0.01, 0.03, 0.031, 0.05])
        windows = list(fixed_windows(events, 0.01))
        self.assertEqual([w.event_count for w in windows], [1, 2, 0, 2, 1])
        indices = [i for w in windows for i in range(w.begin, w.stop)]
        self.assertEqual(indices, list(range(len(events))))
        self.assertTrue(windows[-1].right_closed)
        self.assertTrue(all(not w.right_closed for w in windows[:-1]))

    def test_explicit_end_is_exclusive_and_decimal_boundaries_are_stable(self):
        events = stream([0, 0.1, 0.2, 0.3, 0.4])
        windows = list(fixed_windows(events, 0.1, end_s=0.4))
        self.assertEqual([w.event_count for w in windows], [1, 1, 1, 1])
        self.assertEqual(windows[2].end_s, 0.3)
        self.assertFalse(any(w.right_closed for w in windows))

    def test_partial_window_reference_time(self):
        windows = list(fixed_windows(stream([0, 0.025]), 0.01))
        self.assertTrue(windows[-1].partial)
        self.assertAlmostEqual(windows[-1].reference_s, 0.0225)
        self.assertAlmostEqual(windows[-1].reference_s - windows[-2].reference_s, 0.0075)

    def test_subinterval_and_empty_recording(self):
        windows = list(fixed_windows(stream([0, 0.1, 0.2]), 0.05, start_s=0.05, end_s=0.2))
        self.assertEqual(sum(w.event_count for w in windows), 1)
        self.assertEqual(list(fixed_windows(stream([]), 0.01)), [])
        windows = list(fixed_windows(stream([]), 0.01, end_s=0.03))
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(w.event_count == 0 for w in windows))
        singleton = list(fixed_windows(stream([0]), 0.01))
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton[0].event_count, 1)

    def test_all_three_policies_consume_same_interval(self):
        events = stream(np.linspace(0, 0.099, 100))
        for duration in (0.01, 0.025, 0.05):
            with self.subTest(duration=duration):
                windows = list(fixed_windows(events, duration, end_s=0.1))
                self.assertEqual(sum(w.event_count for w in windows), len(events))

    def test_invalid_intervals(self):
        for options in ({"duration_s": 0}, {"duration_s": float("nan")}, {"duration_s": 0.01, "start_s": -1},
                        {"duration_s": 0.01, "end_s": 0}, {"duration_s": 1e-10, "end_s": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                list(fixed_windows(stream([0, 0.1]), **options))
