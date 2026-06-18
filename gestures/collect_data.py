"""Record a labeled hand-pose dataset from the webcam.

Hold a pose and press its number key to capture the (translation/scale-invariant)
21-landmark vector with that label. Samples append to
data/samples/gesture_poses.csv, which train_pose.py turns into a learned
classifier that replaces the hand-tuned geometric thresholds.

Keys:  1 OPEN_PALM  2 FIST  3 POINT  4 PEACE  5 THUMBS_UP  6 THUMBS_DOWN  7 PINCH
       q quit
Run:   python -m gestures.collect_data
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

from gestures import poses
from vision import draw
from vision.camera import Camera
from vision.hands import HandTracker

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "gesture_poses.csv"
_KEYS = {
    "1": poses.OPEN_PALM, "2": poses.FIST, "3": poses.POINT, "4": poses.PEACE,
    "5": poses.THUMBS_UP, "6": poses.THUMBS_DOWN, "7": poses.PINCH,
}


def _primary(hands):
    return max(hands, key=lambda h: poses.hand_scale(h.landmarks)) if hands else None


def run(camera_index: int = 0) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    fh = open(CSV_PATH, "a", newline="")
    writer = csv.writer(fh)
    if new:
        writer.writerow([f"f{i}" for i in range(63)] + ["label"])

    tracker = HandTracker(num_hands=1)
    counts: dict[str, int] = {}
    print("Collecting hand poses. Press 1-7 to label, 'q' to quit.")

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            hand = _primary(tracker.process(frame))
            vec = None
            if hand is not None:
                draw.draw_hand(frame, hand.landmarks)
                vec = poses.landmark_vector(hand.landmarks)

            draw.panel(frame, 10, 10, 360, 40 + 20 * len(_KEYS))
            draw.text(frame, "Pose collector  (1-7 label, q quit)", 22, 34, draw.WHITE, 0.55)
            yy = 56
            for k, name in _KEYS.items():
                draw.text(frame, f"{k} {name}: {counts.get(name, 0)}", 22, yy, draw.DIM, 0.5)
                yy += 20

            cv2.imshow("Pose collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            ch = chr(key) if key != 255 else ""
            if ch in _KEYS and vec is not None:
                label = _KEYS[ch]
                writer.writerow(list(vec) + [label])
                counts[label] = counts.get(label, 0) + 1

    fh.close()
    tracker.close()
    cv2.destroyAllWindows()
    print(f"\nSaved to {CSV_PATH}\nCounts: {counts}")


if __name__ == "__main__":
    run()
