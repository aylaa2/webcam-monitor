"""Finger cursor control with dwell-to-click.

While the controlling hand is POINTING (index out), the index fingertip drives
the mouse. Hold the cursor still for `dwell_time` seconds AFTER it has moved and
it performs a left click. If the cursor never moved (you just held the pose), it
does NOT click — so you can point without triggering a click.

A small central region of the camera maps to the whole screen so the edges are
easy to reach, and the position is EMA-smoothed for a steady pointer.
"""
from __future__ import annotations

import time


def _pyautogui():
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    return pyautogui


class CursorController:
    def __init__(self, dry_run: bool = False, dwell_time: float = 2.5,
                 move_thresh: float = 16.0, margin: float = 0.18,
                 smooth: float = 0.45) -> None:
        self.dry_run = dry_run
        self.dwell_time = dwell_time
        self.move_thresh = move_thresh     # screen px that counts as "moving"
        self.margin = margin               # central camera fraction mapped to screen
        self.smooth = smooth
        try:
            self._screen = _pyautogui().size()
        except Exception:  # noqa: BLE001
            self._screen = (1440, 900)
        self._sx = self._sy = None         # smoothed screen position
        self._last: tuple[float, float] | None = None
        self._still_since = 0.0
        self._has_moved = False            # gate: don't click unless it actually moved
        self._clicked = False
        self.progress = 0.0                # 0..1 dwell progress (for the HUD ring)
        self.pos: tuple[float, float] | None = None   # normalized cursor pos for HUD

    def reset(self) -> None:
        self._sx = self._sy = None
        self._last = None
        self._has_moved = self._clicked = False
        self.progress = 0.0
        self.pos = None

    def _map(self, v: float) -> float:
        lo = self.margin
        return min(1.0, max(0.0, (v - lo) / (1 - 2 * lo)))

    def update(self, tip_xy, active: bool) -> str | None:
        """Feed the index-tip (normalized x,y) and whether cursor mode is on.
        Returns a label when a dwell-click fires, else None."""
        if not active or tip_xy is None:
            self.reset()
            return None
        sw, sh = self._screen
        tx = self._map(tip_xy[0]) * sw
        ty = self._map(tip_xy[1]) * sh
        if self._sx is None:
            self._sx, self._sy = tx, ty
        else:
            self._sx += self.smooth * (tx - self._sx)
            self._sy += self.smooth * (ty - self._sy)
        x, y = self._sx, self._sy
        self.pos = (x / sw, y / sh)

        if not self.dry_run:
            try:
                _pyautogui().moveTo(int(x), int(y))
            except Exception:  # noqa: BLE001
                pass

        now = time.monotonic()
        if self._last is None:
            self._last = (x, y)
            self._still_since = now
            return None
        dist = ((x - self._last[0]) ** 2 + (y - self._last[1]) ** 2) ** 0.5
        self._last = (x, y)

        action = None
        if dist > self.move_thresh:                 # moving -> reset dwell, mark moved
            self._still_since = now
            self._has_moved = True
            self._clicked = False
            self.progress = 0.0
        elif self._has_moved and not self._clicked:  # still, and it had moved -> dwell
            self.progress = min(1.0, (now - self._still_since) / self.dwell_time)
            if now - self._still_since >= self.dwell_time:
                self._clicked = True
                self.progress = 0.0
                if not self.dry_run:
                    try:
                        _pyautogui().click()
                    except Exception:  # noqa: BLE001
                        pass
                action = "Left click (dwell)"
        else:
            self.progress = 0.0
        return action
