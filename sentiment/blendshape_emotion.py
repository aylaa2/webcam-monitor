"""Rule-based emotion estimate from the 52 ARKit-style blendshapes.

This is the zero-training baseline: it works the instant you run the app, and
it is completely transparent — each emotion is a hand-written weighted sum of
muscle activations (e.g. happy = smile + cheek-squint). It also yields a
continuous valence/arousal reading, which is the affective-computing way to
describe sentiment beyond discrete labels.

You can later swap in the trained SVM (classifier.py) and compare the two.
"""
from __future__ import annotations

import numpy as np

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]


def _g(bs: dict[str, float], *names: str) -> float:
    """Average of the given blendshapes (handles left/right pairs)."""
    vals = [bs.get(n, 0.0) for n in names]
    return float(sum(vals) / len(vals)) if vals else 0.0


def predict(bs: dict[str, float]) -> tuple[dict[str, float], float, float]:
    smile = _g(bs, "mouthSmileLeft", "mouthSmileRight")
    frown = _g(bs, "mouthFrownLeft", "mouthFrownRight")
    brow_down = _g(bs, "browDownLeft", "browDownRight")
    brow_inner = _g(bs, "browInnerUp")
    brow_outer = _g(bs, "browOuterUpLeft", "browOuterUpRight")
    jaw_open = _g(bs, "jawOpen")
    eye_wide = _g(bs, "eyeWideLeft", "eyeWideRight")
    eye_squint = _g(bs, "eyeSquintLeft", "eyeSquintRight")
    cheek_squint = _g(bs, "cheekSquintLeft", "cheekSquintRight")
    mouth_press = _g(bs, "mouthPressLeft", "mouthPressRight")
    mouth_stretch = _g(bs, "mouthStretchLeft", "mouthStretchRight")
    mouth_lower = _g(bs, "mouthLowerDownLeft", "mouthLowerDownRight")
    upper_up = _g(bs, "mouthUpperUpLeft", "mouthUpperUpRight")
    sneer = _g(bs, "noseSneerLeft", "noseSneerRight")

    scores = {
        "happy": 1.2 * smile + 0.5 * cheek_squint,
        "surprise": 0.9 * jaw_open + 0.8 * eye_wide + 0.6 * brow_outer + 0.4 * brow_inner,
        "sad": 1.0 * frown + 0.6 * brow_inner + 0.4 * mouth_lower,
        "angry": 1.0 * brow_down + 0.6 * mouth_press + 0.4 * eye_squint + 0.3 * sneer,
        "fear": 0.8 * eye_wide + 0.7 * brow_inner + 0.6 * mouth_stretch + 0.3 * jaw_open,
        "disgust": 1.0 * sneer + 0.7 * upper_up + 0.3 * brow_down,
        "neutral": 0.18,  # constant floor so a still face reads as neutral
    }
    # Clip negatives and normalize to a probability distribution.
    arr = np.array([max(0.0, scores[e]) for e in EMOTIONS], dtype=np.float32)
    total = arr.sum()
    probs = (arr / total) if total > 0 else np.eye(len(EMOTIONS))[0]
    out = {e: float(p) for e, p in zip(EMOTIONS, probs)}

    valence = float(np.clip(smile - 0.5 * frown - 0.5 * brow_down, -1.0, 1.0))
    arousal = float(np.clip(0.6 * jaw_open + 0.5 * eye_wide + 0.4 * brow_inner, 0.0, 1.0))
    return out, valence, arousal
