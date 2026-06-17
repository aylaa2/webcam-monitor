"""Minimal UI widgets drawn directly into OpenCV windows.

We build the interface as composited frames + native OpenCV windows rather than
a browser or a heavyweight GUI toolkit: it keeps the live-video pipeline intact,
adds zero dependencies, and gives real OS windows you can place side by side.
Buttons are clickable via cv2.setMouseCallback.
"""
from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class MouseState:
    """Captures the last left-click so the main loop can hit-test buttons."""

    def __init__(self) -> None:
        self.click: tuple[int, int] | None = None
        self.hover: tuple[int, int] = (0, 0)

    def __call__(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.hover = (x, y)

    def take_click(self) -> tuple[int, int] | None:
        c = self.click
        self.click = None
        return c


class Button:
    def __init__(self, x, y, w, h, label, color) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.color = color
        self.enabled = True

    def contains(self, pt) -> bool:
        if pt is None:
            return False
        px, py = pt
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, img, hover=(0, 0)) -> None:
        col = self.color if self.enabled else (70, 70, 70)
        if self.enabled and self.contains(hover):
            col = tuple(min(255, int(c * 1.25)) for c in col)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), col, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      (235, 235, 235), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(self.label, _FONT, 0.6, 2)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.label, (tx, ty), _FONT, 0.6, (245, 245, 245), 2, cv2.LINE_AA)


def wrap_text(text: str, width_chars: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def text(img, s, x, y, color=(238, 238, 238), scale=0.5, thick=1) -> None:
    cv2.putText(img, s, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)
