"""Per-person neutral-face calibration.

Blendshape coefficients have large person-to-person resting offsets: some people
rest with a slight smile, raised brows, or a turned head. Measuring emotion from
*raw* blendshapes therefore mislabels neutral faces. Capturing each person's
baseline once at the start and measuring every later frame as a DEVIATION from
it is the single biggest accuracy improvement for this kind of system — and it's
completely transparent (just a per-feature subtraction).
"""
from __future__ import annotations

import time

import numpy as np


class BaselineCalibrator:
    def __init__(self, seconds: float = 2.5, min_frames: int = 15) -> None:
        self.seconds = seconds
        self.min_frames = min_frames
        self._bs_sum: dict[str, float] = {}
        self._yaw: list[float] = []
        self._pitch: list[float] = []
        self._n = 0
        self._t0: float | None = None
        self.baseline_bs: dict[str, float] = {}
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.done = False

    def update(self, blendshapes: dict[str, float], yaw: float, pitch: float) -> None:
        if self.done:
            return
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        for k, v in blendshapes.items():
            self._bs_sum[k] = self._bs_sum.get(k, 0.0) + v
        self._yaw.append(yaw)
        self._pitch.append(pitch)
        self._n += 1
        if now - self._t0 >= self.seconds and self._n >= self.min_frames:
            self._finish()

    def _finish(self) -> None:
        if self._n:
            self.baseline_bs = {k: v / self._n for k, v in self._bs_sum.items()}
            self.base_yaw = float(np.median(self._yaw))
            self.base_pitch = float(np.median(self._pitch))
        self.done = True

    def remaining(self) -> float:
        if self.done:
            return 0.0
        if self._t0 is None:
            return self.seconds
        return max(0.0, self.seconds - (time.monotonic() - self._t0))

    def adjust_blendshapes(self, blendshapes: dict[str, float]) -> dict[str, float]:
        """Return blendshapes as positive deviations above the resting baseline."""
        if not self.baseline_bs:
            return blendshapes
        return {k: max(0.0, v - self.baseline_bs.get(k, 0.0)) for k, v in blendshapes.items()}
