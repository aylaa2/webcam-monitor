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
from collections import deque

import cv2
import numpy as np

from gestures import poses
from gestures.actions import ActionController
from gestures.cursor import CursorController
from gestures.state_machine import GestureRecognizer
from vision import draw, ui
from vision.camera import Camera
from vision.hands import HandTracker

_LM_EMA = 0.5  # landmark smoothing (higher = snappier, lower = steadier)
_LEGEND = [
    ("point index finger", "move cursor"),
    ("hold still ~2.5s", "left click"),
    ("finger circle R / L", "volume up / down"),
    ("open -> fist", "close window"),
    ("peace (hold)", "screenshot"),
    ("thumbs up / down", "play-pause / mute"),
    ("pinch", "mission control"),
    ("two open palms", "switch app"),
]


def _primary_hand(hands):
    if not hands:
        return None
    return max(hands, key=lambda h: poses.hand_scale(h.landmarks))


def _split_hands(hands):
    """Return (activator, command).

    The activator is an OPEN-PALM hand; the command hand is the other one. This
    is independent of which side of the frame each hand is on, so it doesn't
    depend on the camera being mirrored. Returns (None, None) until two hands are
    visible and at least one of them is an open palm.
    """
    if len(hands) < 2:
        return None, None
    tagged = [(h, poses.classify(h.landmarks)) for h in hands]
    palms = [h for h, p in tagged if p == poses.OPEN_PALM]
    if not palms:
        return None, None
    non_palms = [h for h, p in tagged if p != poses.OPEN_PALM]
    activator = palms[0]
    if non_palms:
        command = non_palms[0]               # the hand that's actually gesturing
    else:
        # Both hands are open palms (e.g. the open phase of an open->fist grab).
        # Keep the command hand stable by picking the rightmost as command.
        ordered = sorted(hands, key=lambda hd: float(hd.landmarks[:, 0].mean()))
        activator, command = ordered[0], ordered[-1]
    return activator, command


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
    recognizer = GestureRecognizer()  # defaults tuned for reliability + refractory
    controller = ActionController(dry_run=not live)
    cursor = CursorController(dry_run=not live)

    cv2.namedWindow(MAIN)
    cv2.moveWindow(MAIN, 60, 60)
    log: deque[tuple[str, str]] = deque(maxlen=200)
    total = 0
    log_placed = False

    smoothed_lm = None
    toast_msg, toast_t = "", 0.0
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
            h, w = frame.shape[:2]
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
                    landmarks=smoothed_lm, handedness=cmd.handedness, score=cmd.score)
            else:
                smoothed_lm = None
                hand_in = None

            pose_now = poses.classify(hand_in.landmarks) if hand_in is not None else "-"

            event = recognizer.update(hand_in)   # hand_in is None when not active -> no-op
            if event:
                record(event)

            # --- finger cursor + dwell-click (only while active and POINTING) ---
            cursor_active = active and pose_now == poses.POINT
            tip = ((float(smoothed_lm[8, 0]), float(smoothed_lm[8, 1]))
                   if cursor_active and hand_in is not None else None)
            click = cursor.update(tip, cursor_active)
            if click:
                log_action(click)
            now = time.monotonic()
            if (len(hands) == 2
                    and all(poses.classify(h.landmarks) == poses.OPEN_PALM for h in hands)
                    and now - switch_cooldown > 1.5):
                switch_cooldown = now
                label = controller.dispatch("SWITCH")
                if label:
                    toast_msg, toast_t = label, now

            # --- main HUD ---
            draw.panel(frame, 10, 10, 360, 86)
            draw.text(frame, "Gesture Control", 22, 36, draw.WHITE, 0.7, 2)
            draw.text(frame, banner, 22, 58, draw.RED if live else draw.GREEN, 0.5)
            pose_now = poses.classify(hand_in.landmarks) if hand_in else "—"
            draw.text(frame, f"pose: {pose_now}", 22, 80, draw.DIM, 0.5)
            _draw_legend(frame)
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - fps_t))
            fps_t = now
            draw.text(frame, f"{fps:4.0f} fps", 300, 36, draw.DIM, 0.5)
            if toast_msg and now - toast_t < 1.2:
                draw.toast(frame, toast_msg)

            cv2.imshow(MAIN, frame)
            cv2.imshow(LOG, _render_log(log, total, live))
            if not log_placed:
                cv2.moveWindow(LOG, 60 + 1290, 60)
                log_placed = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    tracker.close()
    cv2.destroyAllWindows()
