"""Regenerate the tiny committed fixture from a deterministic synthetic stream."""

from pathlib import Path

import numpy as np

from adaptive_event_vo.datasets import EventStream, generate_events, save_events


def main() -> None:
    stream = generate_events(seed=7, width=32, height=24)
    indices = np.linspace(0, len(stream) - 1, 320, dtype=int)
    sample = EventStream(stream.t[indices], stream.x[indices], stream.y[indices], stream.p[indices], 32, 24)
    output = Path(__file__).resolve().parents[1] / "examples" / "synthetic_events.csv"
    save_events(sample, output)
    print(f"Saved {len(sample)} synthetic events to {output}")


if __name__ == "__main__":
    main()
