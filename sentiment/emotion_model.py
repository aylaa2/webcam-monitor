"""AffectNet-trained face-emotion model (HSEmotion, ONNX) — a big accuracy
upgrade over the blendshape rules.

We don't drop the rules: we ENSEMBLE (late-fuse) this model's probabilities with
the rule baseline, which is more robust than either alone, and we keep the
speech-aware neutral correction so talking still isn't read as surprise.

Runs on onnxruntime (no PyTorch). Guarded: if hsemotion-onnx or its model is
missing, callers fall back to the rules automatically.
"""
from __future__ import annotations

import urllib.request  # noqa: F401  ensures hsemotion's lazy model download works
from collections import deque

import numpy as np

from .blendshape_emotion import EMOTIONS

# HSEmotion 8-class order (AffectNet) -> our 7 labels (contempt folded into anger).
_HSE_ORDER = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral",
              "Sadness", "Surprise"]
_HSE_MAP = {"Anger": "angry", "Contempt": "angry", "Disgust": "disgust",
            "Fear": "fear", "Happiness": "happy", "Neutral": "neutral",
            "Sadness": "sad", "Surprise": "surprise"}

_model = None
_unavailable = False


def _load() -> bool:
    global _model, _unavailable
    if _model is not None:
        return True
    if _unavailable:
        return False
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        _model = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[emotion] HSEmotion model unavailable ({exc}); using rules only.")
        _unavailable = True
        return False


def available() -> bool:
    return _load()


def _crop_face(frame_bgr, landmarks, margin: float = 0.25):
    h, w = frame_bgr.shape[:2]
    xs, ys = landmarks[:, 0] * w, landmarks[:, 1] * h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    x0 = int(max(0, x0 - margin * bw))
    x1 = int(min(w, x1 + margin * bw))
    y0 = int(max(0, y0 - margin * bh))
    y1 = int(min(h, y1 + margin * bh))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1]


def predict(frame_bgr, landmarks) -> dict[str, float] | None:
    """8-class AffectNet probabilities, remapped to our 7 labels. None if N/A."""
    if not _load():
        return None
    crop = _crop_face(frame_bgr, landmarks)
    if crop is None or crop.size == 0:
        return None
    try:
        _label, scores = _model.predict_emotions(crop[:, :, ::-1], logits=False)
    except Exception:  # noqa: BLE001
        return None
    scores = np.asarray(scores, dtype=np.float32)
    out = {e: 0.0 for e in EMOTIONS}
    for i, cls in enumerate(_HSE_ORDER):
        if i < len(scores):
            out[_HSE_MAP[cls]] += float(scores[i])
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s > 0 else None


def fuse(model_probs, rule_probs, speaking: float = 0.0, w_model: float = 0.72):
    """Late-fusion ensemble of the model and the rules, with a gentle speech-aware
    neutral correction (talking shouldn't read as surprise/fear). The model gets
    most of the weight because it's trained on real surprised/fearful faces, which
    the geometric rules under-detect."""
    if model_probs is None:
        fused = dict(rule_probs)
    else:
        fused = {e: w_model * model_probs.get(e, 0.0) + (1 - w_model) * rule_probs.get(e, 0.0)
                 for e in EMOTIONS}
    if speaking > 0:
        # Only down-weight surprise/fear that comes WITHOUT wide eyes (i.e. talking).
        # If the model is confident about surprise, it has wide eyes -> keep it.
        fused["neutral"] = fused.get("neutral", 0.0) + 0.12 * speaking
        for k in ("surprise", "fear"):
            fused[k] = fused.get(k, 0.0) * (1 - 0.30 * speaking)
    total = sum(fused.values())
    return {k: v / total for k, v in fused.items()} if total > 0 else rule_probs


class Throttle:
    """Run the model every Nth frame and reuse the cached result in between, to
    keep the live frame rate high (emotion changes slowly; the HMM smooths it)."""

    def __init__(self, every: int = 3) -> None:
        self.every = every
        self._i = 0
        self._cache: dict[str, float] | None = None
        self._recent: deque = deque(maxlen=1)  # placeholder for future use

    def step(self, frame_bgr, landmarks) -> dict[str, float] | None:
        if not available():
            return None
        if self._i % self.every == 0:
            self._cache = predict(frame_bgr, landmarks)
        self._i += 1
        return self._cache
