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

import numpy as np

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
# (tip, pip-ish joint) per finger for the extension test.
_FINGERS = {
    "thumb": (4, 2),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}
_PALM = [0, 5, 9, 13, 17]


def palm_center(lm: np.ndarray) -> np.ndarray:
    return lm[_PALM, :2].mean(axis=0)


def hand_scale(lm: np.ndarray) -> float:
    """A stable size estimate: wrist -> middle-finger MCP distance."""
    return float(np.linalg.norm(lm[9, :2] - lm[0, :2]) + 1e-6)


def fingers_extended(lm: np.ndarray, margin: float = 1.07) -> dict[str, bool]:
    c = palm_center(lm)
    out: dict[str, bool] = {}
    for name, (tip, joint) in _FINGERS.items():
        d_tip = np.linalg.norm(lm[tip, :2] - c)
        d_joint = np.linalg.norm(lm[joint, :2] - c)
        out[name] = d_tip > d_joint * margin
    return out


def pinch_distance(lm: np.ndarray) -> float:
    """Thumb-tip to index-tip distance, normalized by hand size."""
    return float(np.linalg.norm(lm[4, :2] - lm[8, :2]) / hand_scale(lm))


def classify(lm: np.ndarray) -> str:
    ext = fingers_extended(lm)
    n = sum(ext.values())

    # Pinch: thumb + index tips touching while the index is still reaching out
    # in front (tip farther from the palm than its PIP). Requiring "not buried"
    # is what separates a pinch from a closed fist, where the index curls in.
    c = palm_center(lm)
    index_buried = np.linalg.norm(lm[8, :2] - c) < np.linalg.norm(lm[6, :2] - c) * 0.95
    if pinch_distance(lm) < 0.30 and not index_buried:
        return PINCH

    if n == 5:
        return OPEN_PALM
    if n == 0:
        return FIST
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return PEACE
    if ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return POINT
    if ext["thumb"] and not ext["index"] and not ext["middle"] and not ext["pinky"]:
        # Thumb-only: up or down based on thumb tip vs wrist (image y grows down).
        return THUMBS_UP if lm[4, 1] < lm[WRIST, 1] else THUMBS_DOWN
    return UNKNOWN


def landmark_vector(lm: np.ndarray) -> np.ndarray:
    """Translation- and scale-invariant 63-d vector for an optional ML model."""
    centered = lm - lm[WRIST]
    centered = centered / hand_scale(lm)
    return centered.flatten().astype(np.float32)
