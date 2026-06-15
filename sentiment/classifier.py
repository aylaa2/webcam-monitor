"""Classic ML emotion classifier over the hand-crafted feature vector.

A scikit-learn Pipeline(StandardScaler -> SVM/RandomForest). This is the
"trainable, still fully explainable" model: you can inspect feature scalings,
RandomForest feature importances, SVM support vectors, etc.

If no model file exists yet, `load()` returns None and the app transparently
falls back to the blendshape rule baseline — so the demo always runs.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "emotion_svm.joblib"


class EmotionClassifier:
    def __init__(self, pipeline, labels: list[str]) -> None:
        self.pipeline = pipeline
        self.labels = labels

    def predict_proba(self, feat: np.ndarray) -> dict[str, float]:
        proba = self.pipeline.predict_proba(feat.reshape(1, -1))[0]
        return {lbl: float(p) for lbl, p in zip(self.labels, proba)}

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "EmotionClassifier | None":
        if not path.exists():
            return None
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["labels"])
