"""Record your own labeled emotion dataset from the webcam.

While looking at the camera and acting an emotion, press the matching number key
to capture the current frame's geometric feature vector with that label. Samples
are appended to data/samples/emotion_features.csv, which train_svm.py consumes.

This makes the project fully self-contained: you build, train on, and evaluate
your *own* data — no external dataset download required.

Keys:  0 neutral  1 happy  2 sad  3 angry  4 surprise  5 fear  6 disgust
       q quit
Run:   python -m sentiment.collect_data
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

from sentiment import features as F
from sentiment.blendshape_emotion import EMOTIONS
from vision import draw
from vision.camera import Camera
from vision.face import FaceTracker

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "emotion_features.csv"
_KEYS = {str(i): EMOTIONS[i] for i in range(len(EMOTIONS))}


def run(camera_index: int = 0) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    fh = open(CSV_PATH, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow([f"f{i}" for i in range(9)] + ["label"])

    tracker = FaceTracker(num_faces=1)
    counts = {e: 0 for e in EMOTIONS}
    last_feat = None

    print("Collecting. Press 0-6 to label the current face, 'q' to quit.")
    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            res = tracker.process(frame)
            if res.ok:
                last_feat = F.compute(res.landmarks, res.transform, w / h)
                draw.draw_face_points(frame, res.landmarks, list(F.LEFT_EYE) +
                                      list(F.RIGHT_EYE) + list(F.MOUTH_CORNERS))

            draw.panel(frame, 10, 10, 360, 40 + 20 * len(EMOTIONS))
            draw.text(frame, "Dataset collector  (0-6 label, q quit)", 22, 34, draw.WHITE, 0.55)
            yy = 56
            for i, e in enumerate(EMOTIONS):
                draw.text(frame, f"{i} {e}: {counts[e]}", 22, yy, draw.DIM, 0.5)
                yy += 20

            cv2.imshow("Dataset collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            ch = chr(key) if key != 255 else ""
            if ch in _KEYS and last_feat is not None:
                label = _KEYS[ch]
                writer.writerow(list(last_feat.feature_vector()) + [label])
                counts[label] += 1

    fh.close()
    tracker.close()
    cv2.destroyAllWindows()
    print(f"\nSaved samples to {CSV_PATH}")
    print("Per-class counts:", counts)


if __name__ == "__main__":
    run()
