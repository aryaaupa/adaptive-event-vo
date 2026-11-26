"""Known translations of event-count patterns, not a physical sensor simulator."""

import numpy as np

from .loader import EventStream


def translated_events(dx: int = 2, dy: int = -1, *, seed: int = 19) -> EventStream:
    """Two 10 ms windows with identical count texture shifted by (dx, dy).

    The fixture specifies image displacement, not camera motion. All features
    have a 20-pixel margin so the tested shifts have no occlusion or clipping.
    """
    if not isinstance(dx, int) or not isinstance(dy, int) or abs(dx) > 10 or abs(dy) > 10:
        raise ValueError("fixture translations must be integer pixels in [-10, 10]")
    rng = np.random.default_rng(seed)
    pattern = rng.random((56, 88)) < 0.18
    y, x = np.nonzero(pattern)
    multiplicity = rng.integers(1, 5, len(x))
    x, y = np.repeat(x + 20, multiplicity), np.repeat(y + 20, multiplicity)
    polarity = rng.choice([-1, 1], len(x))
    time = np.linspace(0, 0.009, len(x))
    return EventStream(np.concatenate([time, 0.01 + time]), np.concatenate([x, x + dx]),
                       np.concatenate([y, y + dy]), np.tile(polarity, 2), 128, 96)
