"""Dynamic gesture recognition via an explicit finite state machine.

Static poses (poses.py) answer "what shape is the hand in *now*". Real commands
are usually *transitions* over time:

  - GRAB        open hand -> fist  (detected by finger COUNT, robust to angle)
  - SWIPE_L/R   fast horizontal wrist travel with an open palm   (prev/next slide)
  - ROTATE_R/L  index finger drawn in a circle                   (volume dial)
  - PINCH       rising edge of a thumb+index pinch
  - THUMBS_UP / THUMBS_DOWN / PEACE   held briefly, fired once

An FSM is deterministic, explainable, and needs zero training data — the kind of
classic architecture that's nice to defend academically.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import poses

# Dynamic event names.
GRAB = "GRAB"
ROTATE_RIGHT = "ROTATE_RIGHT"
ROTATE_LEFT = "ROTATE_LEFT"
PINCH = "PINCH"
THUMBS_UP = "THUMBS_UP"
THUMBS_DOWN = "THUMBS_DOWN"
PEACE = "PEACE"


class GestureRecognizer:
    def __init__(
        self,
        grab_window: float = 1.0,      # max seconds between open hand and fist
        cooldown: float = 1.4,         # min seconds between any two discrete events
        refractory: float = 1.0,       # post-gesture lockout: ignore + clear buffers
        hold_frames: int = 6,          # frames a static gesture must persist
        rot_time: float = 1.3,         # window of the rotation trail
        rot_angle: float = 1.2,        # radians swept to count as one rotation step
        rot_cooldown: float = 0.4,     # seconds between volume steps while rotating
        rot_min_span: float = 0.025,   # min 2D extent so a stray move isn't a turn
    ) -> None:
        self.grab_window = grab_window
        self.cooldown = cooldown
        self.refractory = refractory
        self.hold_frames = hold_frames
        self.rot_time = rot_time
        self.rot_angle = rot_angle
        self.rot_cooldown = rot_cooldown
        self.rot_min_span = rot_min_span

        self._last_open_t = 0.0
        self._rot: deque[tuple[float, float, float]] = deque()   # index-tip trail
        self._last_event_t = 0.0
        self._last_rot_t = 0.0
        self._refractory_until = 0.0
        self._pinch_active = False
        self._hold_pose = ""
        self._hold_count = 0
        self._latched = ""

    def _cooling_down(self, now: float) -> bool:
        return (now - self._last_event_t) < self.cooldown

    def locked(self) -> bool:
        return time.monotonic() < self._refractory_until

    def _fire(self, event: str, now: float) -> str:
        self._last_event_t = now
        self._refractory_until = now + self.refractory
        self._last_open_t = 0.0
        return event

    def _check_rotation(self, now: float) -> str | None:
        if len(self._rot) < 6:
            return None
        pts = np.array([(x, y) for _, x, y in self._rot], dtype=np.float64)
        c = pts.mean(axis=0)
        span = pts.max(axis=0) - pts.min(axis=0)
        if min(span) < self.rot_min_span:        # a straight line, not a circle
            return None
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        d = np.diff(ang)
        d = (d + np.pi) % (2 * np.pi) - np.pi    # unwrap to [-pi, pi]
        total = float(d.sum())
        if abs(total) < self.rot_angle:
            return None
        if float((np.sign(d) == np.sign(total)).mean()) < 0.6:   # not a consistent turn
            return None
        if now - self._last_rot_t < self.rot_cooldown:
            return None
        self._last_rot_t = now
        while len(self._rot) > 4:                 # keep a little trail for the next step
            self._rot.popleft()
        return ROTATE_RIGHT if total > 0 else ROTATE_LEFT

    def update(self, hand) -> str | None:
        now = time.monotonic()
        if hand is None:
            self._rot.clear()
            self._pinch_active = False
            self._hold_count = 0
            self._latched = ""
            return None

        lm = hand.landmarks
        pose = poses.classify(lm)
        n_ext = poses.extended_count(lm)

        # --- ROTATION dial: index circling (own cadence, not refractory-gated) ---
        if pose == poses.POINT:
            self._rot.append((now, float(lm[8, 0]), float(lm[8, 1])))
            while self._rot and now - self._rot[0][0] > self.rot_time:
                self._rot.popleft()
            ev = self._check_rotation(now)
            if ev:
                return ev
        else:
            self._rot.clear()

        # --- refractory lockout for the discrete gestures ---
        if now < self._refractory_until:
            self._last_open_t = 0.0
            self._hold_pose = pose
            self._hold_count = 0
            self._pinch_active = pose == poses.PINCH
            if pose in (poses.THUMBS_UP, poses.THUMBS_DOWN, poses.PEACE):
                self._latched = pose
            return None

        # --- GRAB: open hand (>=3 fingers out) then a fist, by finger COUNT ---
        if n_ext >= 3:
            self._last_open_t = now
        if (pose == poses.FIST and (now - self._last_open_t) < self.grab_window
                and not self._cooling_down(now)):
            self._last_open_t = 0.0
            return self._fire(GRAB, now)

        # --- PINCH: rising edge ---
        is_pinch = pose == poses.PINCH
        if is_pinch and not self._pinch_active and not self._cooling_down(now):
            self._pinch_active = True
            return self._fire(PINCH, now)
        if not is_pinch:
            self._pinch_active = False

        # --- latched static gestures (held briefly, fire once) ---
        static_map = {poses.THUMBS_UP: THUMBS_UP, poses.THUMBS_DOWN: THUMBS_DOWN,
                      poses.PEACE: PEACE}
        if pose == self._hold_pose:
            self._hold_count += 1
        else:
            self._hold_pose = pose
            self._hold_count = 1
            if pose not in static_map:
                self._latched = ""
        if (pose in static_map and self._hold_count >= self.hold_frames
                and self._latched != pose and not self._cooling_down(now)):
            self._latched = pose
            return self._fire(static_map[pose], now)

        return None
