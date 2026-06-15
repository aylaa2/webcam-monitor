"""Small CNN emotion classifier (FER2013) — the deep-learning comparison point.

This is intentionally kept apart from the live interview loop: the project runs
end-to-end with NO deep-learning dependency. Install the extras only to train /
evaluate this model and contrast it with the geometric SVM in your writeup:

    .venv/bin/python -m pip install -r requirements-ml.txt

FER2013 labels (Kaggle order):
    0 angry  1 disgust  2 fear  3 happy  4 sad  5 surprise  6 neutral
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "emotion_cnn.pt"


class EmotionCNN(nn.Module):
    """Compact VGG-style net for 48x48 grayscale faces (~1.3M params)."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FERPredictor:
    """Loads trained weights and predicts emotion probabilities from a face crop."""

    def __init__(self, weights: Path = WEIGHTS_PATH) -> None:
        if not weights.exists():
            raise FileNotFoundError(
                f"Missing {weights.name}. Train it with:  python -m sentiment.train_cnn"
            )
        self.device = torch.device("cpu")
        self.model = EmotionCNN().to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, gray_face_48: np.ndarray) -> dict[str, float]:
        x = torch.from_numpy(gray_face_48.astype(np.float32) / 255.0)
        x = x.view(1, 1, 48, 48).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return {lbl: float(p) for lbl, p in zip(FER_LABELS, probs)}
