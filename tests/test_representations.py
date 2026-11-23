import unittest

import numpy as np

from adaptive_event_vo.datasets import EventStream
from adaptive_event_vo.representations import event_counts, polarity_image


class RepresentationTests(unittest.TestCase):
    def test_counts_preserve_collisions_polarity_and_sensor_orientation(self):
        events = EventStream(np.array([0, 1, 2, 3]), np.array([2, 2, 2, 0]), np.array([1, 1, 1, 0]), np.array([1, 1, -1, -1]), 3, 2)
        counts = event_counts(events)
        self.assertEqual(counts.shape, (2, 2, 3))
        self.assertEqual(counts[1, 1, 2], 2)
        self.assertEqual(counts[0, 1, 2], 1)
        self.assertEqual(counts[0, 0, 0], 1)
        self.assertEqual(counts.sum(), len(events))
        image = polarity_image(events)
        np.testing.assert_allclose(image[1, 2], [0.52, 0.23, 0.65])
        np.testing.assert_allclose(image[0, 1], [1, 1, 1])

    def test_empty_window_produces_white_image_and_zero_counts(self):
        empty = EventStream(*(np.array([]) for _ in range(4)), width=3, height=2)
        self.assertEqual(event_counts(empty).sum(), 0)
        self.assertTrue(np.all(polarity_image(empty) == 1))


if __name__ == "__main__":
    unittest.main()
