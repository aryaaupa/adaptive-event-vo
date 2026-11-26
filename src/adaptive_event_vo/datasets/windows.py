"""Exhaustive fixed-duration scheduling, including silence and the final event."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from collections.abc import Iterator

import numpy as np

from .loader import EventStream


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start_s: float
    end_s: float
    begin: int
    stop: int
    right_closed: bool
    partial: bool

    @property
    def reference_s(self) -> float:
        return (self.start_s + self.end_s) / 2

    @property
    def event_count(self) -> int:
        return self.stop - self.begin


def fixed_windows(
    events: EventStream, duration_s: float, *, start_s: float = 0, end_s: float | None = None,
) -> Iterator[TimeWindow]:
    """Partition a declared [start, end) interval; preserve empty windows.

    With no end, the last timestamp defines the recording endpoint and the last
    window alone is right-closed to include that event exactly once. A singleton
    recording uses one nominal window. Explicit end bounds are always exclusive.
    Midpoints define frame reference times, including the final partial window.
    """
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(start_s) or start_s < 0:
        raise ValueError("duration must be positive and start nonnegative, both finite")
    inferred = end_s is None
    if inferred:
        if not len(events):
            return
        end_s = float(events.t[-1])
        if end_s < start_s:
            raise ValueError("start is after the final event")
        if end_s == start_s:
            end_s = float(Decimal(str(start_s)) + Decimal(str(duration_s)))
    if not np.isfinite(end_s) or end_s <= start_s:
        raise ValueError("end must be finite and greater than start")
    start, end, step = map(lambda x: Decimal(str(x)), (start_s, end_s, duration_s))
    number = int(((end - start) / step).to_integral_value(rounding=ROUND_CEILING))
    if number > 1_000_000:
        raise ValueError("requested interval exceeds one million windows")
    begin = int(np.searchsorted(events.t, start_s, side="left"))
    for index in range(number):
        left = start + index * step
        nominal_right = start + (index + 1) * step
        right = min(nominal_right, end)
        left_s, right_s = float(left), float(right)
        if right_s <= left_s:
            raise ValueError("window duration is below timestamp precision at this offset")
        closed = inferred and index == number - 1
        stop = int(np.searchsorted(events.t, right_s, side="right" if closed else "left"))
        yield TimeWindow(index, left_s, right_s, begin, stop, closed, right < nominal_right)
        begin = stop
