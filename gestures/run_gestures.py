"""Gesture control studio — webcam gestures drive real macOS actions, with a
separate native "Action Log" window recording every action that fired.

Two windows:
  - "Gesture Control" : live camera, hand skeletons, gesture legend, current pose.
  - "Action Log"      : timestamped, scrolling history of every action performed.

Accuracy aids: EMA-smoothed hand landmarks + hold-to-confirm + per-action cooldown.

Controls:  q or ESC = quit
Run:
    python app.py gestures            # LIVE — controls macOS
    python app.py gestures --dry-run  # safe preview, prints/logs only
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

MAIN, LOG = "Gesture Control", "Action Log"
_LM_EMA = 0.5
_LEGEND = [
    ("point index finger", "move cursor"),
    ("hold still ~2.5s", "left click"),
    ("finger circle R / L", "volume up / down"),
    ("open -> fist", "close window"),
    ("peace (hold)", "screenshot"),
    ("thumbs up / down", "play-pause / mute"),
    ("pinch", "mission control"),
]


def _primary_hand(hands):
    if not hands:
        return None
    return max(hands, key=lambda h: poses.hand_scale(h.landmarks))


def _draw_legend(frame):
    x = frame.shape[1] - 300
    draw.panel(frame, x, 10, 290, 30 + 22 * len(_LEGEND))
    draw.text(frame, "LEFT palm on -> RIGHT hand:", x + 12, 32, draw.WHITE, 0.55, 2)
    y = 54
    for g, a in _LEGEND:
        draw.text(frame, g, x + 12, y, draw.YELLOW, 0.45)
        draw.text(frame, a, x + 165, y, draw.DIM, 0.45)
        y += 22


def _render_log(entries: deque, total: int, live: bool) -> np.ndarray:
    width, rows = 440, 18
    canvas = np.full((70 + rows * 26, width, 3), (30, 28, 26), np.uint8)
    ui.text(canvas, "ACTION LOG", 20, 34, (120, 200, 140), 0.8, 2)
    mode = "LIVE" if live else "DRY-RUN"
    ui.text(canvas, f"{mode}   total: {total}", 20, 56, (165, 165, 160), 0.5)
    cv2.line(canvas, (20, 66), (width - 20, 66), (70, 68, 64), 1)
    y = 92
    for ts, label in list(entries)[-rows:][::-1]:
        ui.text(canvas, ts, 20, y, (150, 180, 235), 0.5)
        ui.text(canvas, label, 110, y, (238, 238, 238), 0.5)
        y += 26
    if not entries:
        ui.text(canvas, "make a gesture to begin...", 20, 92, (120, 120, 118), 0.5)
    return canvas


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
    switch_cooldown = 0.0
    fps_t, fps = time.monotonic(), 0.0

    banner = "LIVE - controlling macOS" if live else "DRY-RUN - logging only"
    print(f"Gesture Control open.  [{banner}]   q/ESC to quit.")
    if live:
        print("  (window/slide/play actions need Terminal in Accessibility permissions)")

    def log_action(label):
        nonlocal total, toast_msg, toast_t
        log.append((time.strftime("%H:%M:%S"), label))
        toast_msg, toast_t = label, time.monotonic()
        total += 1

    def record(event):
        label = controller.dispatch(event)
        if label:
            log_action(label)
            return True
        return False

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            hands = tracker.process(frame)
            for hnd in hands:
                draw.draw_hand(frame, hnd.landmarks)

            # --- two-hand activation gate ---------------------------------
            # Nothing happens unless your LEFT hand is an open palm. Then your
            # RIGHT hand makes the command gesture. This prevents incidental hand
            # movements (scratching your face, etc.) from doing anything.
            left_hand = right_hand = None
            if len(hands) >= 2:                 # mirrored view: left hand = smaller x
                ordered = sorted(hands, key=lambda hd: float(hd.landmarks[:, 0].mean()))
                left_hand, right_hand = ordered[0], ordered[-1]
            active = (left_hand is not None
                      and poses.classify(left_hand.landmarks) == poses.OPEN_PALM)
            cmd = right_hand if active else None

            if cmd is not None:
                if smoothed_lm is None or smoothed_lm.shape != cmd.landmarks.shape:
                    smoothed_lm = cmd.landmarks.copy()
                smoothed_lm = _LM_EMA * cmd.landmarks + (1 - _LM_EMA) * smoothed_lm
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

            # highlight the activating left palm
            if left_hand is not None:
                lx = int(left_hand.landmarks[:, 0].mean() * w)
                ly = int(left_hand.landmarks[:, 1].mean() * h)
                col = draw.GREEN if active else (90, 90, 90)
                cv2.circle(frame, (lx, ly), 26, col, 2, cv2.LINE_AA)

            # --- cursor pointer + dwell ring (drawn on the fingertip) ---
            if cursor_active and hand_in is not None:
                cx, cy = int(smoothed_lm[8, 0] * w), int(smoothed_lm[8, 1] * h)
                cv2.circle(frame, (cx, cy), 10, draw.YELLOW, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, draw.WHITE, -1, cv2.LINE_AA)
                if cursor.progress > 0:
                    cv2.ellipse(frame, (cx, cy), (16, 16), -90, 0,
                                int(360 * cursor.progress), draw.GREEN, 3, cv2.LINE_AA)

            # --- main HUD ---
            draw.panel(frame, 10, 10, 360, 86)
            draw.text(frame, "Gesture Control", 22, 36, draw.WHITE, 0.7, 2)
            draw.text(frame, banner, 22, 58, draw.RED if live else draw.GREEN, 0.5)
            if active:
                draw.text(frame, f"ACTIVE   pose: {pose_now}", 22, 80, draw.GREEN, 0.5)
                if recognizer.locked():
                    draw.text(frame, "WAIT", 300, 80, draw.RED, 0.5, 2)
            else:
                draw.text(frame, "show your LEFT palm to activate", 22, 80, draw.YELLOW, 0.5)
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
