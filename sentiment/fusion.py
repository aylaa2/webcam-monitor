"""Transparent multi-cue fusion into a single 'engagement' score.

Instead of a black box, engagement is an explicit weighted blend of four
interpretable cues an interviewer actually reacts to:

  facing      head pointed at the camera        (attention)
  eye_contact gaze centered on the camera
  positivity  affective valence                 (warm vs flat)
  composure   not blinking nervously fast

Every weight is visible and defensible. Swap the weights for a learned logistic
regression later if you want a data-driven version to compare against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .features import FaceFeatures

WEIGHTS = {"facing": 0.35, "eye_contact": 0.25, "positivity": 0.25, "composure": 0.15}


@dataclass
class Engagement:
    facing: float
    eye_contact: float
    positivity: float
    composure: float
    overall: float


def _gauss(x: float, sigma: float) -> float:
    return math.exp(-(x * x) / (2.0 * sigma * sigma))


def compute(feat: FaceFeatures, valence: float, blink_rate: float) -> Engagement:
    facing = _gauss(feat.yaw, 18.0) * _gauss(feat.pitch, 18.0)
    gaze = _gauss(feat.gaze_x, 0.30) * _gauss(feat.gaze_y, 0.35)
    # Real eye contact needs BOTH the head pointed at the camera AND the gaze
    # centered, so glancing away (by moving the eyes OR turning the head) drops it.
    eye_contact = facing * gaze
    positivity = (valence + 1.0) / 2.0
    composure = 1.0 - min(1.0, max(0.0, (blink_rate - 25.0) / 40.0))

    overall = (
        WEIGHTS["facing"] * facing
        + WEIGHTS["eye_contact"] * eye_contact
        + WEIGHTS["positivity"] * positivity
        + WEIGHTS["composure"] * composure
    )
    return Engagement(facing, eye_contact, positivity, composure, overall)
