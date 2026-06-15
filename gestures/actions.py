"""Map gesture events to real macOS actions.

Each action uses the most *reliable* mechanism for that job rather than forcing
everything through one API:
  - screenshot      -> `screencapture` CLI        (rock solid, no focus needed)
  - volume / mute   -> AppleScript (osascript)     (no Accessibility needed)
  - window / slides / mission-control / switch / play-pause -> pyautogui keys
    (these are app-context, so they need Accessibility permission for Terminal)

dry_run=True prints actions without performing them (safe preview).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pyautogui

from . import state_machine as sm

pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0.0

SHOTS_DIR = Path.home() / "Desktop"


def _osa(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ActionController:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._map = {
            sm.GRAB: ("Close window  (Cmd+W)", lambda: pyautogui.hotkey("command", "w")),
            sm.SWIPE_RIGHT: ("Next slide  (->)", lambda: pyautogui.press("right")),
            sm.SWIPE_LEFT: ("Previous slide  (<-)", lambda: pyautogui.press("left")),
            sm.SWIPE_UP: ("Volume up", lambda: self._volume(+12)),
            sm.SWIPE_DOWN: ("Volume down", lambda: self._volume(-12)),
            sm.PINCH: ("Mission Control", lambda: pyautogui.hotkey("ctrl", "up")),
            sm.THUMBS_UP: ("Play / Pause", lambda: pyautogui.press("playpause")),
            sm.THUMBS_DOWN: ("Mute toggle", self._mute),
            sm.PEACE: ("Screenshot", self._screenshot),
            "SWITCH": ("Switch app  (Cmd+Tab)", lambda: pyautogui.hotkey("command", "tab")),
        }

    def label(self, event: str) -> str:
        entry = self._map.get(event)
        return entry[0] if entry else event

    def dispatch(self, event: str) -> str | None:
        entry = self._map.get(event)
        if not entry:
            return None
        label, fn = entry
        tag = "DRY-RUN" if self.dry_run else "ACTION "
        print(f"[{tag}] {event:<12} -> {label}")
        if not self.dry_run:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] action failed: {exc}")
        return label

    # --- concrete actions ---
    def _volume(self, delta: int) -> None:
        _osa(
            f"set v to (output volume of (get volume settings)) + {delta}\n"
            "if v > 100 then set v to 100\n"
            "if v < 0 then set v to 0\n"
            "set volume output volume v"
        )

    def _mute(self) -> None:
        _osa("set volume output muted (not (output muted of (get volume settings)))")

    def _screenshot(self) -> None:
        out = SHOTS_DIR / f"gesture_{time.strftime('%Y%m%d_%H%M%S')}.png"
        subprocess.run(["screencapture", "-x", str(out)], check=False)
        print(f"        saved -> {out}")
