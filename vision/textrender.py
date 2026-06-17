"""Unicode-capable text rendering for OpenCV images.

cv2.putText uses Hershey fonts that only cover ASCII, so accented characters
(Romanian ă â î ș ț, etc.) render as '?'. We batch text and draw it with Pillow
using a TrueType font that supports full Unicode — fixing the diacritics in the
report / detail windows. Text is collected and drawn in one pass per image so
the PIL<->numpy conversion happens only once.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATH: str | None = None
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font_path() -> str:
    global _FONT_PATH
    if _FONT_PATH:
        return _FONT_PATH
    try:
        from matplotlib import font_manager
        _FONT_PATH = font_manager.findfont("DejaVu Sans")  # ships with matplotlib
    except Exception:  # noqa: BLE001
        _FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    return _FONT_PATH


def _font(px: int) -> ImageFont.FreeTypeFont:
    f = _FONT_CACHE.get(px)
    if f is None:
        f = ImageFont.truetype(_font_path(), px)
        _FONT_CACHE[px] = f
    return f


def scale_to_px(scale: float) -> int:
    """Map a cv2 fontScale to a TrueType pixel size (HERSHEY ~ 30*scale tall)."""
    return max(9, int(round(scale * 30)))


class TextLayer:
    """Collect text items (in cv2-baseline coordinates), then render them all at
    once onto a BGR image with a Unicode font."""

    def __init__(self) -> None:
        self.items: list[tuple[str, int, int, tuple, int, bool]] = []

    def put(self, text, x, baseline_y, color_bgr=(238, 238, 238), scale=0.5,
            bold=False) -> None:
        self.items.append((str(text), int(x), int(baseline_y), color_bgr,
                           scale_to_px(scale), bool(bold)))

    def render(self, img_bgr: np.ndarray) -> np.ndarray:
        if not self.items:
            return img_bgr
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for text, x, by, (b, g, r), px, bold in self.items:
            font = _font(px)
            ascent, _descent = font.getmetrics()
            top = by - ascent
            d.text((x, top), text, font=font, fill=(r, g, b))
            if bold:
                d.text((x + 1, top), text, font=font, fill=(r, g, b))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
