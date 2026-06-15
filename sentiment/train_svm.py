"""Train the classic emotion classifier on your collected feature data.

Compares an SVM (RBF) against a RandomForest with cross-validation, prints a
confusion matrix and the RandomForest feature importances (the explainability
payoff), then saves the better model to models/emotion_svm.joblib so the
interview demo picks it up automatically.

Run:  python -m sentiment.train_svm
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sentiment.classifier import EmotionClassifier

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "emotion_features.csv"
FEATURE_NAMES = ["ear", "mouth_ar", "smile", "brow_raise", "gaze_x", "gaze_y",
                 "pitch", "yaw", "roll"]


def _load() -> tuple[np.ndarray, np.ndarray]:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"No data at {CSV_PATH}.\nRun:  python -m sentiment.collect_data  first."
        )
    X, y = [], []
    with open(CSV_PATH) as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 10:
                continue
            X.append([float(v) for v in row[:9]])
            y.append(row[9])
    return np.array(X, dtype=np.float32), np.array(y)


def main() -> None:
    X, y = _load()
    classes = sorted(set(y))
    print(f"Loaded {len(X)} samples across {len(classes)} classes: {classes}")
    if len(X) < 20 or len(classes) < 2:
        raise SystemExit("Need more data — collect at least ~20 samples over 2+ emotions.")

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=4.0, gamma="scale",
                                              probability=True))
    rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=300, random_state=0))

    for name, model in (("SVM (RBF)", svm), ("RandomForest", rf)):
        scores = cross_val_score(model, X, y, cv=min(5, len(X) // len(classes)))
        print(f"{name:14s} CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit final SVM on a train split and show a confusion matrix.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y if len(X) >= 4 * len(classes) else None)
    svm.fit(Xtr, ytr)
    pred = svm.predict(Xte)
    print("\nSVM held-out report:\n", classification_report(yte, pred, zero_division=0))
    print("Confusion matrix (rows=true):")
    print("labels:", list(svm.classes_))
    print(confusion_matrix(yte, pred, labels=svm.classes_))

    # RandomForest feature importances — the interpretability talking point.
    rf.fit(X, y)
    importances = rf.named_steps["randomforestclassifier"].feature_importances_
    print("\nFeature importances (RandomForest):")
    for nm, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1]):
        print(f"  {nm:11s} {imp:.3f}")

    # Save the SVM refit on ALL data for deployment.
    svm.fit(X, y)
    EmotionClassifier(svm, list(svm.classes_)).save()
    print("\nSaved -> models/emotion_svm.joblib  (interview mode will use it next run)")


if __name__ == "__main__":
    main()
