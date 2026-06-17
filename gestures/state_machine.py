"""Dynamic gesture recognition via an explicit finite state machine.

Static poses (poses.py) answer "what shape is the hand in *now*". Real commands
are usually *transitions* over time:

  - GRAB        OPEN_PALM -> FIST  within a short window   (your "close window")
  - SWIPE_*     fast wrist travel while the hand is open
  - PINCH       rising edge of a thumb+index pinch
  - THUMBS_UP / THUMBS_DOWN / PEACE   held briefly, fired once (latched)

An FSM is deliberately chosen over a neural net here: it is deterministic,
trivially explainable, and needs zero training data — exactly the kind of
classic architecture that's nice to defend academically. (An optional LSTM
upgrade is described in the README.)
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import poses

# Dynamic event names.
GRAB = "GRAB"
SWIPE_LEFT = "SWIPE_LEFT"
SWIPE_RIGHT = "SWIPE_RIGHT"
SWIPE_UP = "SWIPE_UP"
SWIPE_DOWN = "SWIPE_DOWN"
PINCH = "PINCH"
THUMBS_UP = "THUMBS_UP"
THUMBS_DOWN = "THUMBS_DOWN"
PEACE = "PEACE"


class GestureRecognizer:
    def __init__(
        self,
        grab_window: float = 0.8,      # max seconds between open palm and fist
        swipe_dist: float = 0.28,      # min normalized wrist travel for a swipe
        swipe_time: float = 0.35,      # window over which travel is measured
        cooldown: float = 1.6,         # min seconds between any two fired events
        refractory: float = 1.1,       # post-gesture lockout: ignore + clear buffers
        hold_frames: int = 7,          # frames a static gesture must persist
    ) -> None:
        self.grab_window = grab_window
        self.swipe_dist = swipe_dist
        self.swipe_time = swipe_time
        self.cooldown = cooldown
        self.refractory = refractory
        self.hold_frames = hold_frames

        self._last_open_t = 0.0
        self._traj: deque[tuple[float, float, float]] = deque()  # (t, x, y)
        self._last_event_t = 0.0
        self._refractory_until = 0.0
        self._pinch_active = False
        self._hold_pose = ""
        self._hold_count = 0
        self._latched = ""  # last latched static gesture, prevents re-fire

    def _cooling_down(self, now: float) -> bool:
        return (now - self._last_event_t) < self.cooldown

    def locked(self) -> bool:
        """True while in the post-gesture refractory window (for the HUD)."""
        return time.monotonic() < self._refractory_until

    def _fire(self, event: str, now: float) -> str:
        self._last_event_t = now
        self._refractory_until = now + self.refractory
        self._traj.clear()
        self._last_open_t = 0.0
        return event

    def update(self, hand) -> str | None:
        """Feed the current detected hand (or None). Returns an event or None."""
        now = time.monotonic()

        if hand is None:
            self._traj.clear()
            self._pinch_active = False
            self._hold_count = 0
            self._latched = ""
            return None

        lm = hand.landmarks
        pose = poses.classify(lm)
        wrist = lm[poses.WRIST, :2]

        # --- refractory lockout: after a gesture fires, ignore input briefly and
        #     keep buffers clean so the hand returning to rest can't re-trigger ---
        if now < self._refractory_until:
            self._traj.clear()
            self._last_open_t = 0.0
            self._hold_pose = pose
            self._hold_count = 0
            self._pinch_active = pose == poses.PINCH
            # block an immediate static re-fire of whatever is being held
            if pose in (poses.THUMBS_UP, poses.THUMBS_DOWN, poses.PEACE):
                self._latched = pose
            return None

        # --- maintain a short wrist trajectory for swipes ---
        self._traj.append((now, float(wrist[0]), float(wrist[1])))
        while self._traj and now - self._traj[0][0] > self.swipe_time:
            self._traj.popleft()

        # --- GRAB: remember when the palm was last open, then watch for a fist ---
        if pose == poses.OPEN_PALM:
            self._last_open_t = now
        if (
            pose == poses.FIST
            and (now - self._last_open_t) < self.grab_window
            and not self._cooling_down(now)
        ):
            self._last_open_t = 0.0
            return self._fire(GRAB, now)

        # --- SWIPE: large net wrist displacement with a clearly open palm ---
        if pose == poses.OPEN_PALM and len(self._traj) >= 3:
            t0, x0, y0 = self._traj[0]
            dx, dy = wrist[0] - x0, wrist[1] - y0
            if not self._cooling_down(now):
                if abs(dx) > self.swipe_dist and abs(dx) > abs(dy):
                    self._traj.clear()
                    return self._fire(SWIPE_RIGHT if dx > 0 else SWIPE_LEFT, now)
                if abs(dy) > self.swipe_dist and abs(dy) > abs(dx):
                    self._traj.clear()
                    return self._fire(SWIPE_DOWN if dy > 0 else SWIPE_UP, now)

        # --- PINCH: rising edge ---
        is_pinch = pose == poses.PINCH
        if is_pinch and not self._pinch_active and not self._cooling_down(now):
            self._pinch_active = True
            return self._fire(PINCH, now)
        if not is_pinch:
            self._pinch_active = False

        # --- latched static gestures (held briefly, fire once) ---
        static_map = {
            poses.THUMBS_UP: THUMBS_UP,
            poses.THUMBS_DOWN: THUMBS_DOWN,
            poses.PEACE: PEACE,
        }
        if pose == self._hold_pose:
            self._hold_count += 1
        else:
            self._hold_pose = pose
            self._hold_count = 1
            if pose not in static_map:
                self._latched = ""
        if (
            pose in static_map
            and self._hold_count >= self.hold_frames
            and self._latched != pose
            and not self._cooling_down(now)
        ):
            self._latched = pose
            return self._fire(static_map[pose], now)

        return None
