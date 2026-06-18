"""Train a learned hand-pose classifier on your collected landmark data.

Replaces the hand-tuned geometric thresholds in poses.py with a RandomForest on
the 63-d landmark vector. Doubles as the evaluation harness: prints cross-
validated accuracy, a held-out classification report, and a confusion matrix, so
"more accurate" is measurable. Saves models/gesture_pose.joblib (poses.classify
picks it up automatically).

Run:  python -m gestures.train_pose
"""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "gesture_poses.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_pose.joblib"


def _load():
    if not CSV_PATH.exists():
        raise SystemExit(f"No data at {CSV_PATH}. Run: python -m gestures.collect_data")
    X, y = [], []
    with open(CSV_PATH) as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 64:
                continue
            X.append([float(v) for v in row[:63]])
            y.append(row[63])
    return np.array(X, dtype=np.float32), np.array(y)


def main() -> None:
    X, y = _load()
    classes = sorted(set(y))
    print(f"Loaded {len(X)} samples across {len(classes)} poses: {classes}")
    if len(X) < 20 or len(classes) < 2:
        raise SystemExit("Collect at least ~20 samples over 2+ poses first.")

    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=300, random_state=0))
    cv = min(5, np.min(np.unique(y, return_counts=True)[1]))
    if cv >= 2:
        scores = cross_val_score(model, X, y, cv=cv)
        print(f"Cross-val accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y if len(X) >= 4 * len(classes) else None)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("\nHeld-out report:\n", classification_report(yte, pred, zero_division=0))
    print("Confusion matrix (rows=true):")
    print("labels:", list(model.classes_))
    print(confusion_matrix(yte, pred, labels=model.classes_))

    model.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "labels": list(model.classes_)}, MODEL_PATH)
    print(f"\nSaved -> {MODEL_PATH}  (poses.classify will use it automatically)")


if __name__ == "__main__":
    main()
