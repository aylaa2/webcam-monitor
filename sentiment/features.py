"""Hand-crafted facial features from the 478 landmarks + head-pose matrix.

These are the *explainable* signals an interview analysis cares about:
  - eye openness / blink   (Eye Aspect Ratio)
  - smile / mouth openness (geometric ratios)
  - eyebrow raise
  - gaze direction         (iris position within the eye)
  - head pose              (pitch / yaw / roll, for "is the subject engaged?")

`feature_vector()` packs the geometric ratios into a fixed vector so the SVM /
RandomForest in classifier.py can be trained on them — the classic, fully
inspectable alternative to a CNN.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# --- landmark index groups (MediaPipe FaceMesh, 478-point iris model) ---
LEFT_EYE = (33, 160, 158, 133, 153, 144)    # p1..p6 for EAR
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_EYE_CORNERS = (33, 133)                 # outer, inner
RIGHT_EYE_CORNERS = (263, 362)
LEFT_IRIS = 468
RIGHT_IRIS = 473
MOUTH_CORNERS = (61, 291)
MOUTH_VERT = (13, 14)                        # inner upper, inner lower lip
LEFT_BROW, LEFT_EYE_TOP = 105, 159
RIGHT_BROW, RIGHT_EYE_TOP = 334, 386


@dataclass
class FaceFeatures:
    ear: float            # mean eye aspect ratio (0=closed, ~0.3 open)
    mouth_ar: float       # mouth open ratio
    smile: float          # mouth-width / interocular  (bigger => smiling)
    brow_raise: float     # brow-to-eye distance, normalized
    gaze_x: float         # -1 left .. 0 center .. +1 right
    gaze_y: float         # -1 up .. 0 .. +1 down
    pitch: float          # head tilt up/down (deg)
    yaw: float            # head turn left/right (deg)
    roll: float           # head roll (deg)

    def feature_vector(self) -> np.ndarray:
        return np.array(
            [self.ear, self.mouth_ar, self.smile, self.brow_raise,
             self.gaze_x, self.gaze_y, self.pitch / 90.0, self.yaw / 90.0,
             self.roll / 90.0],
            dtype=np.float32,
        )


def _p(lm, i, aspect):
    """2D point with x scaled by aspect so distances are isotropic."""
    return np.array([lm[i, 0] * aspect, lm[i, 1]], dtype=np.float32)


def _ear(lm, idx, aspect) -> float:
    p1, p2, p3, p4, p5, p6 = (_p(lm, i, aspect) for i in idx)
    horiz = np.linalg.norm(p1 - p4) + 1e-6
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    return float(vert / (2.0 * horiz))


def head_pose(transform: np.ndarray | None) -> tuple[float, float, float]:
    """Decompose the 4x4 facial transform into (pitch, yaw, roll) degrees."""
    if transform is None:
        return 0.0, 0.0, 0.0
    rot = transform[:3, :3].astype(np.float64)
    angles, *_ = cv2.RQDecomp3x3(rot)  # returns degrees about x, y, z
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    return pitch, yaw, roll


def compute(lm: np.ndarray, transform, aspect: float) -> FaceFeatures:
    interocular = np.linalg.norm(_p(lm, 33, aspect) - _p(lm, 263, aspect)) + 1e-6

    ear = 0.5 * (_ear(lm, LEFT_EYE, aspect) + _ear(lm, RIGHT_EYE, aspect))

    mc_l, mc_r = _p(lm, MOUTH_CORNERS[0], aspect), _p(lm, MOUTH_CORNERS[1], aspect)
    mv_t, mv_b = _p(lm, MOUTH_VERT[0], aspect), _p(lm, MOUTH_VERT[1], aspect)
    mouth_w = np.linalg.norm(mc_l - mc_r)
    mouth_h = np.linalg.norm(mv_t - mv_b)
    mouth_ar = float(mouth_h / (mouth_w + 1e-6))
    smile = float(mouth_w / interocular)

    brow_l = np.linalg.norm(_p(lm, LEFT_BROW, aspect) - _p(lm, LEFT_EYE_TOP, aspect))
    brow_r = np.linalg.norm(_p(lm, RIGHT_BROW, aspect) - _p(lm, RIGHT_EYE_TOP, aspect))
    brow_raise = float((brow_l + brow_r) / (2.0 * interocular))

    gaze_x, gaze_y = _gaze(lm, aspect)
    pitch, yaw, roll = head_pose(transform)

    return FaceFeatures(ear, mouth_ar, smile, brow_raise, gaze_x, gaze_y,
                        pitch, yaw, roll)


def _gaze(lm, aspect) -> tuple[float, float]:
    """Iris position within each eye, averaged. ~0 = looking at camera."""
    def eye_gaze(iris, corners, top_idx, bot_idx):
        outer = _p(lm, corners[0], aspect)
        inner = _p(lm, corners[1], aspect)
        c = _p(lm, iris, aspect)
        # horizontal: project iris between the two corners -> [-1, +1]
        span = inner - outer
        denom = np.dot(span, span) + 1e-6
        t = np.dot(c - outer, span) / denom            # 0..1
        gx = (t - 0.5) * 2.0
        top = _p(lm, top_idx, aspect)
        bot = _p(lm, bot_idx, aspect)
        h = (bot[1] - top[1]) + 1e-6
        gy = ((c[1] - top[1]) / h - 0.5) * 2.0
        return gx, gy

    lx, ly = eye_gaze(LEFT_IRIS, LEFT_EYE_CORNERS, 159, 145)
    rx, ry = eye_gaze(RIGHT_IRIS, RIGHT_EYE_CORNERS, 386, 374)
    return float((lx + rx) / 2.0), float((ly + ry) / 2.0)


class BlinkCounter:
    """Counts blinks (EAR falling edge) and reports blink rate per minute."""

    def __init__(self, threshold: float = 0.18, min_frames: int = 2) -> None:
        self.threshold = threshold
        self.min_frames = min_frames
        self._below = 0
        self.count = 0

    def update(self, ear: float) -> bool:
        blinked = False
        if ear < self.threshold:
            self._below += 1
        else:
            if self._below >= self.min_frames:
                self.count += 1
                blinked = True
            self._below = 0
        return blinked

    def rate_per_min(self, elapsed_s: float) -> float:
        if elapsed_s <= 0:
            return 0.0
        return self.count / (elapsed_s / 60.0)


class SpeakingDetector:
    """Estimate whether the mouth is *talking* vs holding an expression.

    Talking makes the jaw oscillate open/closed; a held emotion does not. We
    measure the short-window standard deviation of jawOpen — high variance =>
    speaking. Returns a continuous [0,1] 'speaking' level used to discount the
    open-mouth contribution to emotion.
    """

    def __init__(self, window: int = 20, floor: float = 0.02, scale: float = 0.07) -> None:
        self.buf: deque[float] = deque(maxlen=window)
        self.floor = floor
        self.scale = scale

    def update(self, jaw_open: float) -> float:
        self.buf.append(float(jaw_open))
        if len(self.buf) < 8:
            return 0.0
        std = float(np.std(np.fromiter(self.buf, dtype=np.float32)))
        return float(np.clip((std - self.floor) / self.scale, 0.0, 1.0))
