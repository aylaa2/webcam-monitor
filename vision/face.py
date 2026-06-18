"""MediaPipe Face Landmarker wrapper (Tasks API).

Gives us, per frame:
  - 478 3D face landmarks (468 mesh + 10 iris) in normalized [0,1] coords
  - 52 blendshape coefficients (smile, jawOpen, browDown, eyeBlink, ...)
  - a 4x4 facial transformation matrix => head pose, for free

All of this is computed on-device by a small bundled neural net. We treat it as
a *feature extractor*; the actual emotion decision happens downstream in
sentiment/ using classic, explainable methods.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@dataclass
class FaceResult:
    landmarks: np.ndarray | None = None          # (478, 3) normalized x,y,z
    blendshapes: dict[str, float] = field(default_factory=dict)
    transform: np.ndarray | None = None          # (4, 4) face->camera, or None
    n_faces: int = 0                             # number of faces detected this frame

    @property
    def ok(self) -> bool:
        return self.landmarks is not None


class FaceTracker:
    def __init__(self, num_faces: int = 1) -> None:
        if not _MODEL.exists():
            raise FileNotFoundError(
                f"Missing {_MODEL.name}. Run:  python download_models.py"
            )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.monotonic()

    def _timestamp_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def process(self, frame_bgr: np.ndarray) -> FaceResult:
        rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(mp_image, self._timestamp_ms())

        if not res.face_landmarks:
            return FaceResult()

        n_faces = len(res.face_landmarks)
        lms = res.face_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)

        blendshapes: dict[str, float] = {}
        if res.face_blendshapes:
            for cat in res.face_blendshapes[0]:
                blendshapes[cat.category_name] = float(cat.score)

        transform = None
        if res.facial_transformation_matrixes:
            transform = np.array(res.facial_transformation_matrixes[0], dtype=np.float32)

        return FaceResult(landmarks=landmarks, blendshapes=blendshapes,
                          transform=transform, n_faces=n_faces)

    def close(self) -> None:
        self._landmarker.close()
