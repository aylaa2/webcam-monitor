"""Threaded webcam capture.

Grabbing frames on a background thread decouples camera I/O from the (heavier)
inference + drawing work, which keeps the perceived frame rate high. The main
loop always reads the *latest* frame and never blocks waiting on the camera.
"""
from __future__ import annotations

import threading
import time

import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. On macOS, grant the terminal "
                "Camera permission in System Settings > Privacy & Security."
            )
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()
        # Wait briefly for the first frame so callers don't get None.
        for _ in range(50):
            if self._frame is not None:
                break
            time.sleep(0.02)

    def _reader(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        """Return the most recent frame (BGR), already mirrored for a natural
        'selfie' view."""
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is not None:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        self._running = False
        self._t.join(timeout=1.0)
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
