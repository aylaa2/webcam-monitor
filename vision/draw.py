"""Lightweight OpenCV overlay helpers shared by both demos."""
from __future__ import annotations

import cv2
import numpy as np

WHITE = (245, 245, 245)
DIM = (180, 180, 180)
GREEN = (90, 220, 120)
RED = (90, 90, 235)
BLUE = (235, 180, 90)
YELLOW = (90, 210, 235)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Hand skeleton connections (pairs of landmark indices).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """Draw a translucent dark panel for HUD text."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def text(img, s, x, y, color=WHITE, scale=0.6, thick=1) -> None:
    cv2.putText(img, s, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color=GREEN, label: str | None = None) -> None:
    """Horizontal progress bar in [0,1]."""
    frac = max(0.0, min(1.0, float(frac)))
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)
    if label:
        text(img, label, x, y - 4, DIM, 0.45)


def draw_hand(img, landmarks: np.ndarray, color=YELLOW) -> None:
    h, w = img.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
    for a, b in HAND_EDGES:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, WHITE, -1, cv2.LINE_AA)


def draw_face_points(img, landmarks: np.ndarray, indices=None, color=BLUE) -> None:
    h, w = img.shape[:2]
    idx = range(len(landmarks)) if indices is None else indices
    for i in idx:
        p = landmarks[i]
        cv2.circle(img, (int(p[0] * w), int(p[1] * h)), 1, color, -1, cv2.LINE_AA)


def toast(img, msg: str, color=GREEN) -> None:
    """Big transient message in the lower third (e.g. a fired action)."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(msg, _FONT, 1.1, 2)
    x = (w - tw) // 2
    y = int(h * 0.82)
    panel(img, x - 20, y - th - 16, tw + 40, th + 28, 0.6)
    text(img, msg, x, y, color, 1.1, 2)
