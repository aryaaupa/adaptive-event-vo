"""Count-based event frames with explicit collision handling."""

import numpy as np

from adaptive_event_vo.datasets.loader import EventStream


def event_counts(events: EventStream) -> np.ndarray:
    """Return [2, H, W] integer counts: channel 0 negative, channel 1 positive."""
    counts = np.zeros((2, events.height, events.width), dtype=np.int64)
    np.add.at(counts, ((events.p > 0).astype(np.intp), events.y, events.x), 1)
    return counts


def polarity_image(events: EventStream) -> np.ndarray:
    """RGB occupancy: blue positive, orange negative, purple both, white empty.

    This encodes polarity presence; use event_counts for multiplicity. Equal
    opposite-polarity events never cancel out of the displayed image.
    """
    negative, positive = event_counts(events) > 0
    image = np.ones((events.height, events.width, 3), dtype=np.float64)
    image[positive] = (0.08, 0.40, 0.85)
    image[negative] = (0.95, 0.40, 0.10)
    image[positive & negative] = (0.52, 0.23, 0.65)
    return image
