"""Static hand-pose classification from 21 landmarks — pure geometry, no ML.

The key trick is orientation-invariant finger-extension detection: a fingertip
is "extended" if it is farther from the palm center than that finger's middle
joint. This works whether the hand is upright, sideways, or tilted, and every
decision is fully explainable (great for the writeup).

`landmark_vector()` additionally produces a translation/scale-invariant feature
vector, so you can later train a k-NN or RandomForest pose classifier and
compare it against these rules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_pose.joblib"
_model = None
_model_loaded = False

# Pose name constants.
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
POINT = "POINT"
PEACE = "PEACE"
THUMBS_UP = "THUMBS_UP"
THUMBS_DOWN = "THUMBS_DOWN"
PINCH = "PINCH"
UNKNOWN = "UNKNOWN"

WRIST = 0
# (tip, pip) per finger for the wrist-relative extension test.
_FINGER_JOINTS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
_PALM = [0, 5, 9, 13, 17]
_MCPS = [5, 9, 13, 17]


def palm_center(lm: np.ndarray) -> np.ndarray:
    return lm[_PALM, :2].mean(axis=0)


def hand_scale(lm: np.ndarray) -> float:
    """A stable size estimate: wrist -> middle-finger MCP distance."""
    return float(np.linalg.norm(lm[9, :2] - lm[0, :2]) + 1e-6)


def _d(lm, a, b) -> float:
    return float(np.linalg.norm(lm[a, :2] - lm[b, :2]))


def finger_extended(lm: np.ndarray, tip: int, pip: int, margin: float = 1.05) -> bool:
    """A finger is extended if its tip is farther from the WRIST than its PIP
    joint. Wrist-relative distances are robust to hand rotation/orientation, which
    the old palm-centre test was not (that's why the fist barely registered)."""
    return _d(lm, tip, WRIST) > _d(lm, pip, WRIST) * margin


def extended_count(lm: np.ndarray) -> int:
    """How many of the four fingers (index..pinky) are extended (0=fist, 4=open)."""
    return sum(finger_extended(lm, t, p) for t, p in _FINGER_JOINTS.values())


def thumb_extended(lm: np.ndarray) -> bool:
    return _d(lm, 4, WRIST) > _d(lm, 2, WRIST) * 1.10


def pinch_distance(lm: np.ndarray) -> float:
    """Thumb-tip to index-tip distance, normalized by hand size."""
    return float(np.linalg.norm(lm[4, :2] - lm[8, :2]) / hand_scale(lm))


def _classify_geometric(lm: np.ndarray) -> str:
    n = extended_count(lm)
    idx = finger_extended(lm, 8, 6)
    mid = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pky = finger_extended(lm, 20, 18)
    th = thumb_extended(lm)

    # Pinch: thumb + index tips touching while the index isn't buried in the palm.
    c = palm_center(lm)
    index_buried = np.linalg.norm(lm[8, :2] - c) < np.linalg.norm(lm[6, :2] - c) * 0.95
    if pinch_distance(lm) < 0.32 and not index_buried and n <= 2:
        return PINCH

    # Thumbs up / down: fingers curled, thumb out, thumb tip clearly above/below
    # the knuckles. Lenient on the finger count so it's easy to trigger.
    if n <= 1 and th:
        mcp_y = float(lm[_MCPS, 1].mean())
        if lm[4, 1] < mcp_y - 0.04 and lm[4, 1] < lm[WRIST, 1]:
            return THUMBS_UP
        if lm[4, 1] > mcp_y + 0.04 and lm[4, 1] > lm[WRIST, 1]:
            return THUMBS_DOWN

    if n == 0:
        return FIST
    if n >= 4:
        return OPEN_PALM
    if idx and mid and not ring and not pky:
        return PEACE
    if idx and not mid and not ring and not pky:
        return POINT
    return UNKNOWN


def landmark_vector(lm: np.ndarray) -> np.ndarray:
    """Translation- and scale-invariant 63-d vector for an optional ML model."""
    centered = lm - lm[WRIST]
    centered = centered / hand_scale(lm)
    return centered.flatten().astype(np.float32)


def _get_model():
    """Lazily load a trained pose classifier (gestures/train_pose.py) if present."""
    global _model, _model_loaded
    if not _model_loaded:
        _model_loaded = True
        if _MODEL_PATH.exists():
            try:
                import joblib
                _model = joblib.load(_MODEL_PATH)
            except Exception:  # noqa: BLE001
                _model = None
    return _model


def classify(lm: np.ndarray, min_conf: float = 0.6) -> str:
    """Learned classifier (if a trained model exists) with a confidence gate,
    otherwise the geometric rules. Train one with `python -m gestures.train_pose`."""
    model = _get_model()
    if model is not None:
        try:
            proba = model["pipeline"].predict_proba(landmark_vector(lm).reshape(1, -1))[0]
            j = int(np.argmax(proba))
            if proba[j] >= min_conf:
                return model["labels"][j]
        except Exception:  # noqa: BLE001
            pass
    return _classify_geometric(lm)
