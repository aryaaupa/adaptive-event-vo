"""Sparse pyramidal Lucas–Kanade on fixed-scale log-count event images."""

from dataclasses import dataclass

import cv2
import numpy as np

from adaptive_event_vo.datasets.loader import EventStream
from adaptive_event_vo.datasets.windows import TimeWindow


@dataclass(frozen=True)
class FlowConfig:
    max_corners: int = 300
    quality_level: float = 0.01
    min_distance_px: float = 5.0
    block_size: int = 7
    lk_window_px: int = 21
    pyramid_levels: int = 3
    fb_threshold_px: float = 1.0
    max_lk_error: float = 30.0
    min_eigenvalue: float = 1e-4
    min_tracks: int = 8
    count_clip: int = 4
    blur_sigma_px: float = 0.8

    def __post_init__(self):
        integers = (self.max_corners, self.block_size, self.lk_window_px, self.pyramid_levels, self.min_tracks, self.count_clip)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in integers):
            raise ValueError("corner, window, pyramid, track and count settings must be integers")
        if self.max_corners < self.min_tracks or self.min_tracks < 1 or self.pyramid_levels < 0 or self.count_clip < 1:
            raise ValueError("invalid feature budget, pyramid levels or count clip")
        if any(x < 3 or x % 2 == 0 for x in (self.block_size, self.lk_window_px)):
            raise ValueError("feature and LK windows must be odd and at least 3 pixels")
        floats = (self.quality_level, self.min_distance_px, self.fb_threshold_px, self.max_lk_error, self.min_eigenvalue, self.blur_sigma_px)
        if not np.all(np.isfinite(floats)) or any(x <= 0 for x in floats) or self.quality_level > 1:
            raise ValueError("flow thresholds must be finite and positive; quality_level <= 1")


@dataclass(frozen=True)
class FlowResult:
    source_xy: np.ndarray
    target_xy: np.ndarray
    valid: np.ndarray
    fb_error_px: np.ndarray
    lk_error: np.ndarray
    dt_s: float
    status: str

    @property
    def detected(self) -> int:
        return len(self.valid)

    @property
    def accepted(self) -> int:
        return int(self.valid.sum())

    @property
    def displacement_px(self) -> np.ndarray:
        return self.target_xy[self.valid] - self.source_xy[self.valid]


def tracking_frame(events: EventStream, window: TimeWindow, config: FlowConfig) -> np.ndarray:
    """Count both polarities without cancellation; fixed saturation across frames.

    Polarity is intentionally pooled for this image-motion baseline. Counts are
    clipped, log-compressed to uint8, then blurred. No per-frame normalization.
    """
    counts = np.zeros((events.height, events.width), dtype=np.int64)
    selection = slice(window.begin, window.stop)
    np.add.at(counts, (events.y[selection], events.x[selection]), 1)
    image = np.rint(np.log1p(np.minimum(counts, config.count_clip)) * (255 / np.log1p(config.count_clip))).astype(np.uint8)
    return cv2.GaussianBlur(image, (0, 0), config.blur_sigma_px)


def estimate_flow(previous: np.ndarray, current: np.ndarray, *, dt_s: float, config: FlowConfig = FlowConfig()) -> FlowResult:
    """Redetect corners in each source frame, track forward and verify backward.

    Accepted tracks require both LK statuses, finite in-image endpoints, adequate
    local texture (LK minEigThreshold), bounded photometric error in both
    directions, and a bounded forward/backward round-trip pixel distance.
    These checks are confidence filters, not ground-truth accuracy measurements.
    """
    if previous.ndim != 2 or current.shape != previous.shape or previous.dtype != np.uint8 or current.dtype != np.uint8:
        raise ValueError("flow images must be same-size two-dimensional uint8 arrays")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("frame reference-time interval must be finite and positive")

    def empty(status):
        return FlowResult(np.empty((0, 2)), np.empty((0, 2)), np.zeros(0, bool), np.empty(0), np.empty(0), dt_s, status)

    if not np.any(previous):
        return empty("empty_source")
    if not np.any(current):
        return empty("empty_target")
    border = max(3, config.block_size // 2)
    if min(previous.shape) <= 2 * border:
        return empty("image_too_small")
    mask = np.zeros_like(previous)
    mask[border:-border, border:-border] = 255
    points = cv2.goodFeaturesToTrack(previous, config.max_corners, config.quality_level, config.min_distance_px, mask=mask, blockSize=config.block_size)
    if points is None:
        return empty("no_features")
    options = dict(winSize=(config.lk_window_px, config.lk_window_px), maxLevel=config.pyramid_levels,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), minEigThreshold=config.min_eigenvalue)
    forward, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    source = points.reshape(-1, 2)
    target = np.full_like(source, np.nan)
    fb_error, lk_error = np.full(len(source), np.inf), np.full(len(source), np.inf)
    valid = np.zeros(len(source), dtype=bool)
    if forward is not None:
        target[:] = forward.reshape(-1, 2)
        candidates = np.flatnonzero(forward_status.ravel().astype(bool) & np.isfinite(target).all(axis=1))
        if len(candidates):
            backward, backward_status, backward_error = cv2.calcOpticalFlowPyrLK(current, previous, forward[candidates], None, **options)
            if backward is not None:
                fb_error[candidates] = np.linalg.norm(backward.reshape(-1, 2) - source[candidates], axis=1)
                lk_error[candidates] = np.maximum(forward_error.ravel()[candidates], backward_error.ravel())
                valid[candidates] = backward_status.ravel().astype(bool)
    height, width = previous.shape
    valid &= np.isfinite(target).all(axis=1) & np.isfinite(fb_error) & np.isfinite(lk_error)
    valid &= (target[:, 0] >= border) & (target[:, 0] < width - border) & (target[:, 1] >= border) & (target[:, 1] < height - border)
    valid &= (fb_error <= config.fb_threshold_px) & (lk_error <= config.max_lk_error)
    status = "ok" if valid.sum() >= config.min_tracks else "insufficient_tracks"
    return FlowResult(source, target, valid, fb_error, lk_error, dt_s, status)
