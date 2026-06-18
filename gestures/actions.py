"""Map gesture events to real OS actions — cross-platform (Windows + macOS).

Each action uses the most *reliable* mechanism for the current OS:

  Windows
    - volume / mute / play-pause -> native media virtual-keys via ctypes
      (no extra deps; Chrome/Edge honour the media key for YouTube)
    - window / task-view / switch -> pyautogui hotkeys (Ctrl+W, Win+Tab, Alt+Tab)
    - screenshot -> pyautogui.screenshot().save(...)  (needs Pillow, already a dep)

  macOS (original behaviour)
    - screenshot      -> `screencapture` CLI
    - volume / mute   -> AppleScript (osascript)
    - window / slides / mission-control / switch / play-pause -> pyautogui keys

dry_run=True prints actions without performing them (safe preview).
"""
from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

import pyautogui

from . import state_machine as sm

pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0.0

_SYS = platform.system()
_IS_WIN = _SYS == "Windows"
_IS_MAC = _SYS == "Darwin"

# Where screenshots land. Desktop if it exists, else the home folder.
SHOTS_DIR = Path.home() / "Desktop"
if not SHOTS_DIR.exists():
    SHOTS_DIR = Path.home()


# ----------------------------------------------------------------------------
# Windows native media keys (ctypes -> keybd_event). These drive the *system*
# media session, which Chrome/Edge register for the active YouTube tab.
# ----------------------------------------------------------------------------
if _IS_WIN:
    import ctypes

    _VK = {"vol_up": 0xAF, "vol_down": 0xAE, "mute": 0xAD, "play_pause": 0xB3}
    _KEYEVENTF_KEYUP = 0x0002

    def _tap_vk(vk: int) -> None:
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)


def _osa(script: str) -> None:
    """Run an AppleScript snippet (macOS only)."""
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
            "SWITCH": ("Switch app", self._switch_app),
        }
        if dry_run:
            print(f"[actions] platform={_SYS}  (DRY-RUN: nothing will be controlled)")
        else:
            print(f"[actions] platform={_SYS}")

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

    # --- concrete actions -------------------------------------------------
    def _volume(self, direction: int) -> None:
        if _IS_WIN:
            # Each tap is ~2 percentage points; 4 taps = a clear step.
            vk = _VK["vol_up"] if direction > 0 else _VK["vol_down"]
            for _ in range(4):
                _tap_vk(vk)
        elif _IS_MAC:
            delta = 8 * direction
            _osa(
                f"set v to (output volume of (get volume settings)) + {delta}\n"
                "if v > 100 then set v to 100\n"
                "if v < 0 then set v to 0\n"
                "set volume output volume v"
            )

    def _mute(self) -> None:
        if _IS_WIN:
            _tap_vk(_VK["mute"])
        elif _IS_MAC:
            _osa("set volume output muted (not (output muted of (get volume settings)))")

    def _screenshot(self) -> None:
        out = SHOTS_DIR / f"gesture_{time.strftime('%Y%m%d_%H%M%S')}.png"
        if _IS_MAC:
            subprocess.run(["screencapture", "-x", str(out)], check=False)
        else:
            # Windows / Linux: capture via pyautogui (Pillow-backed).
            pyautogui.screenshot().save(str(out))
        print(f"        saved -> {out}")
