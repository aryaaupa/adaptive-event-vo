from dataclasses import replace
import unittest

import cv2
import numpy as np

from adaptive_event_vo.datasets.motion_fixture import translated_events
from adaptive_event_vo.datasets.windows import fixed_windows
from adaptive_event_vo.motion import FlowConfig, estimate_flow, tracking_frame


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

    def pair(self, dx, dy):
        events = translated_events(dx, dy)
        return [tracking_frame(events, w, FlowConfig()) for w in fixed_windows(events, 0.01, end_s=0.02)]

    def test_independently_known_event_image_translations(self):
        for dx, dy in ((2, -1), (-3, 2), (0, 0), (1, 3)):
            with self.subTest(displacement=(dx, dy)):
                a, b = self.pair(dx, dy)
                result = estimate_flow(a, b, dt_s=0.01)
                self.assertEqual(result.status, "ok")
                self.assertGreaterEqual(result.accepted, 20)
                epe = np.linalg.norm(result.displacement_px - [dx, dy], axis=1)
                self.assertLess(np.mean(epe), 0.2)
                self.assertLess(np.percentile(epe, 95), 0.5)

    def test_subpixel_translation_and_time_units(self):
        a, _ = self.pair(0, 0)
        transform = np.array([[1, 0, 1.5], [0, 1, -0.75]], dtype=np.float32)
        b = cv2.warpAffine(a, transform, (a.shape[1], a.shape[0]))
        result = estimate_flow(a, b, dt_s=0.025)
        self.assertEqual(result.status, "ok")
        np.testing.assert_allclose(np.median(result.displacement_px, axis=0), [1.5, -0.75], atol=0.15)
        np.testing.assert_allclose(np.median(result.displacement_px, axis=0) / result.dt_s, [60, -30], atol=6)

    def test_missing_texture_is_failure_and_not_zero_motion(self):
        a, _ = self.pair(0, 0)
        for source, target, expected in ((np.zeros_like(a), a, "empty_source"), (a, np.zeros_like(a), "empty_target"),
                                          (np.full_like(a, 128), np.full_like(a, 128), "no_features")):
            with self.subTest(expected=expected):
                result = estimate_flow(source, target, dt_s=0.01)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.accepted, 0)

    def test_track_budget_is_a_failure_gate(self):
        a, b = self.pair(2, -1)
        result = estimate_flow(a, b, dt_s=0.01, config=FlowConfig(max_corners=300, min_tracks=300))
        self.assertEqual(result.status, "insufficient_tracks")

    def test_invalid_frames_timing_and_settings(self):
        a, b = self.pair(0, 0)
        for previous, current, dt in ((a.astype(float), b, 0.01), (a, b[:-1], 0.01), (a, b, 0), (a, b, float("nan"))):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                estimate_flow(previous, current, dt_s=dt)
        for changes in ({"lk_window_px": 4}, {"min_tracks": 0}, {"count_clip": 0}, {"quality_level": 2}, {"fb_threshold_px": float("nan")}, {"max_corners": 3.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(FlowConfig(), **changes)
