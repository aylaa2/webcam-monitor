"""Gesture mode: webcam -> hand landmarks -> pose -> FSM -> real macOS action.

Accuracy aids:
  - EMA smoothing of the controlling hand's landmarks (kills jitter)
  - hold-to-confirm on static gestures + per-action cooldown (no accidental fires)

Run:
    python app.py gestures            # LIVE — actually controls macOS
    python app.py gestures --dry-run  # safe preview, prints actions only
    python app.py gestures --camera 1
"""
from __future__ import annotations

import time
import types

import cv2
import numpy as np

from gestures import poses
from gestures.actions import ActionController
from gestures.state_machine import GestureRecognizer
from vision import draw
from vision.camera import Camera
from vision.hands import HandTracker

_LM_EMA = 0.5  # landmark smoothing (higher = snappier, lower = steadier)
_LEGEND = [
    ("open palm -> fist", "close window"),
    ("swipe L / R", "prev / next slide"),
    ("swipe up / down", "volume"),
    ("peace (hold)", "screenshot"),
    ("thumbs up / down", "play-pause / mute"),
    ("pinch", "mission control"),
    ("two open palms", "switch app"),
]


def _primary_hand(hands):
    if not hands:
        return None
    return max(hands, key=lambda h: poses.hand_scale(h.landmarks))


def _draw_legend(frame):
    x = frame.shape[1] - 300
    draw.panel(frame, x, 10, 290, 30 + 22 * len(_LEGEND))
    draw.text(frame, "Gestures", x + 12, 32, draw.WHITE, 0.6, 2)
    y = 54
    for g, a in _LEGEND:
        draw.text(frame, g, x + 12, y, draw.YELLOW, 0.45)
        draw.text(frame, a, x + 165, y, draw.DIM, 0.45)
        y += 22


def run(camera_index: int = 0, live: bool = True) -> None:
    tracker = HandTracker(num_hands=2)
    recognizer = GestureRecognizer(cooldown=1.2, hold_frames=6)
    controller = ActionController(dry_run=not live)

    smoothed_lm = None
    toast_msg, toast_t = "", 0.0
    switch_cooldown = 0.0
    fps_t, fps = time.monotonic(), 0.0

    banner = "LIVE — controlling macOS" if live else "DRY-RUN — printing only"
    print(f"Gesture mode ready.  [{banner}]")
    if live:
        print("  (window/slide/play actions need Terminal in Accessibility permissions)")
    print("Press 'q' to quit.")

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            hands = tracker.process(frame)
            for h in hands:
                draw.draw_hand(frame, h.landmarks)

            primary = _primary_hand(hands)
            # EMA-smooth the controlling hand's landmarks for stable poses.
            if primary is not None:
                if smoothed_lm is None or smoothed_lm.shape != primary.landmarks.shape:
                    smoothed_lm = primary.landmarks.copy()
                smoothed_lm = _LM_EMA * primary.landmarks + (1 - _LM_EMA) * smoothed_lm
                hand_in = types.SimpleNamespace(
                    landmarks=smoothed_lm, handedness=primary.handedness, score=primary.score)
            else:
                smoothed_lm = None
                hand_in = None

            event = recognizer.update(hand_in)
            if event:
                label = controller.dispatch(event)
                if label:
                    toast_msg, toast_t = label, time.monotonic()

            now = time.monotonic()
            if (len(hands) == 2
                    and all(poses.classify(h.landmarks) == poses.OPEN_PALM for h in hands)
                    and now - switch_cooldown > 1.5):
                switch_cooldown = now
                label = controller.dispatch("SWITCH")
                if label:
                    toast_msg, toast_t = label, now

            # --- HUD ---
            draw.panel(frame, 10, 10, 360, 86)
            draw.text(frame, "Gesture control", 22, 36, draw.WHITE, 0.7, 2)
            draw.text(frame, banner, 22, 58, draw.RED if live else draw.GREEN, 0.5)
            pose_now = poses.classify(hand_in.landmarks) if hand_in else "—"
            draw.text(frame, f"pose: {pose_now}", 22, 80, draw.DIM, 0.5)
            _draw_legend(frame)

            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - fps_t))
            fps_t = now
            draw.text(frame, f"{fps:4.0f} fps", 300, 36, draw.DIM, 0.5)

            if toast_msg and now - toast_t < 1.2:
                draw.toast(frame, toast_msg)

            cv2.imshow("Gesture control  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    tracker.close()
    cv2.destroyAllWindows()
