"""MediaPipe Hand Landmarker wrapper (Tasks API).

Per frame, for up to `num_hands`, returns 21 landmarks and the handedness
("Left"/"Right"). Landmark indexing follows MediaPipe's hand model:

    0  wrist
    1-4   thumb   (cmc, mcp, ip, tip)
    5-8   index   (mcp, pip, dip, tip)
    9-12  middle
    13-16 ring
    17-20 pinky
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@dataclass
class Hand:
    landmarks: np.ndarray   # (21, 3) normalized x,y,z
    handedness: str         # "Left" or "Right" (as seen in the mirrored image)
    score: float


class HandTracker:
    def __init__(self, num_hands: int = 2) -> None:
        if not _MODEL.exists():
            raise FileNotFoundError(
                f"Missing {_MODEL.name}. Run:  python download_models.py"
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()

    def _timestamp_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def process(self, frame_bgr: np.ndarray) -> list[Hand]:
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(mp_image, self._timestamp_ms())

        hands: list[Hand] = []
        if not res.hand_landmarks:
            return hands
        for i, lms in enumerate(res.hand_landmarks):
            landmarks = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            label, score = "Right", 1.0
            if res.handedness and i < len(res.handedness):
                cat = res.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            hands.append(Hand(landmarks=landmarks, handedness=label, score=score))
        return hands

    def close(self) -> None:
        self._landmarker.close()
