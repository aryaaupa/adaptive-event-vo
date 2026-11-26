"""DAVIS pinhole/radial-tangential calibration metadata; no pose estimation."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]

    @classmethod
    def from_davis_text(cls, text: str, *, width: int, height: int) -> "CameraCalibration":
        values = np.asarray([float(v) for v in text.split()])
        if values.shape != (9,) or not np.all(np.isfinite(values)):
            raise ValueError("DAVIS calibration requires nine finite values: fx fy cx cy k1 k2 p1 p2 k3")
        if width <= 0 or height <= 0 or values[0] <= 0 or values[1] <= 0:
            raise ValueError("sensor size and focal lengths must be positive")
        if not (0 <= values[2] < width and 0 <= values[3] < height):
            raise ValueError("calibration principal point lies outside the sensor")
        return cls(width, height, *map(float, values[:4]), tuple(map(float, values[4:])))

    def to_dict(self) -> dict:
        return {**asdict(self), "model": "pinhole_radtan", "distortion_order": ["k1", "k2", "p1", "p2", "k3"]}
