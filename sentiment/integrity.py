"""Interview integrity — a HEURISTIC estimate of behavioural signals that *can*
accompany off-screen reading or outside help.

This is NOT a cheating detector and proves nothing. It surfaces, transparently,
patterns an interviewer might want to follow up on:
  - frequently looking away from the camera
  - gaze held consistently in one off-screen direction (possible reading)
  - frequent downward gaze (possible notes)
  - very monotone, hesitation-free delivery (possible scripted/read answers)
  - a second face appearing in frame

Every flag is explainable and has innocent explanations too (a second monitor
showing the questions, nervousness, neurodivergence, reading the on-screen
prompt). Always present the disclaimer and let a human judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DISCLAIMER = ("Heuristic only - not proof of cheating. Many flags have innocent "
              "causes (second monitor, nerves, reading the on-screen question).")


@dataclass
class IntegrityReport:
    score: float = 0.0                       # 0-100, higher = more signals
    label: str = "not enough data"
    flags: list[tuple[str, str]] = field(default_factory=list)


# Per-frame "looking away" thresholds (baseline-relative). Combines HEAD POSE
# (the reliable signal for glancing down at a phone / up at a monitor / to a side
# screen) with iris gaze. A run that lasts >= MIN_EVENT_S counts as one event, so
# even brief 1-2 s glances are caught (fraction-of-total metrics missed them).
PITCH_THR = 10.0   # degrees of head tilt up/down
YAW_THR = 12.0     # degrees of head turn left/right
GAZE_THR = 0.16    # normalized iris displacement
MIN_EVENT_S = 0.4

_DIR_LABEL = {
    "down": "down (phone / notes)", "up": "up (monitor above)",
    "right": "to the right (side screen)", "left": "to the left (side screen)",
    "away": "away from the screen",
}


def compute(rec, prosody, words, second_face: bool) -> IntegrityReport:
    n = len(rec.t)
    if n < 10:
        return IntegrityReport()

    t = np.array(rec.t, dtype=np.float64)
    gx = np.array(rec.gaze_x, dtype=np.float32)
    gy = np.array(rec.gaze_y, dtype=np.float32)
    yaw = np.array(rec.yaw, dtype=np.float32)
    pitch = np.array(getattr(rec, "pitch", [0.0] * n) or [0.0] * n, dtype=np.float32)
    if len(pitch) != n:
        pitch = np.zeros(n, dtype=np.float32)

    dt = float((t[-1] - t[0]) / max(1, n - 1)) if n > 1 else 0.04
    min_frames = max(3, int(round(MIN_EVENT_S / max(1e-3, dt))))

    away = ((np.abs(pitch) > PITCH_THR) | (np.abs(yaw) > YAW_THR) |
            (np.abs(gx) > GAZE_THR) | (np.abs(gy) > GAZE_THR))
    off_ratio = float(away.mean())

    # group consecutive away-frames into discrete look-away EVENTS
    events: list[tuple[str, float]] = []
    i = 0
    while i < n:
        if not away[i]:
            i += 1
            continue
        j = i
        while j < n and away[j]:
            j += 1
        if j - i >= min_frames:
            mgx, mgy = float(gx[i:j].mean()), float(gy[i:j].mean())
            mp, my = float(pitch[i:j].mean()), float(yaw[i:j].mean())
            if max(abs(mgx), abs(mgy)) < 0.08 and max(abs(mp), abs(my)) > 6:
                direction = "right" if my > 0 else "left" if my < 0 else "away"
            elif abs(mgy) >= abs(mgx):
                direction = "down" if mgy > 0 else "up"
            else:
                direction = "right" if mgx > 0 else "left"
            events.append((direction, (j - i) * dt))
        i = j

    from collections import Counter
    dirs = Counter(d for d, _ in events)
    n_events = len(events)
    total_away_s = off_ratio * float(t[-1] - t[0])

    monotone = bool(prosody.available and 0 < prosody.f0_range < 30)
    low_filler = bool(words.total_words > 15 and words.filler_rate < 1.5)
    scripted = monotone and low_filler

    score = 0.0
    flags: list[tuple[str, str]] = []

    if n_events:
        score += min(48.0, 15.0 * n_events)
        desc = "; ".join(f"{c}x {_DIR_LABEL.get(d, d)}" for d, c in dirs.most_common())
        flags.append(("Looked away from the screen",
                      f"{n_events} look-away event(s) [{desc}], total ~{total_away_s:.0f}s"))
    if off_ratio > 0.40:
        score += 16
        flags.append(("Often off the screen",
                      f"Eyes/head were off the screen {off_ratio * 100:.0f}% of the time"))
    if scripted:
        score += 16
        flags.append(("Scripted-sounding delivery",
                      "Monotone with almost no hesitations - possible read answers"))
    if second_face:
        score += 45
        flags.append(("Second person in frame",
                      "A second face was detected at some point during the recording"))

    score = min(100.0, score)
    label = ("low concern" if score < 25 else
             "moderate - worth a follow-up" if score < 55 else
             "elevated - review recommended")
    return IntegrityReport(score=score, label=label, flags=flags)
