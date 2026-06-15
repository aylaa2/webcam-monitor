"""Download the MediaPipe Tasks model bundles into ./models.

These are small (~3-8 MB each) and are fetched ONCE. After this, the whole
project runs fully offline — nothing leaves your machine at runtime.

Usage:
    python download_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = {
    # Face mesh + 52 blendshapes + 4x4 facial transformation matrix (head pose).
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    # 21 hand keypoints + handedness.
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

MODELS_DIR = Path(__file__).parent / "models"


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, blocks * block_size * 100 // total)
    sys.stdout.write(f"\r    {pct:3d}%")
    sys.stdout.flush()


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}")
        try:
            urllib.request.urlretrieve(url, dest, _progress)
            print(f"\r    done -> {dest}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[fail] could not download {name}: {exc}")
            return 1
    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
