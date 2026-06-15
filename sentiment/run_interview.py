"""Interview mode: webcam -> face landmarks -> features -> emotion + engagement.

Accuracy pipeline per frame:
    face mesh + 52 blendshapes + head-pose matrix
      -> 2.5s personal NEUTRAL CALIBRATION (measure baseline once)
      -> blendshapes as deviations from baseline  (+ EMA smoothing)
      -> emotion estimate (trained SVM if present, else blendshape rules)
      -> HMM temporal smoothing
      -> baseline-corrected head pose -> multi-cue engagement fusion
      -> live HUD + recording

Quit with 'q' to print the sentiment report, the written description, the peak
emotional moments, and save an annotated graph + text to ./reports.

Run:  python app.py interview   [--camera N]
"""
from __future__ import annotations

import time
from dataclasses import replace

import cv2

from sentiment import blendshape_emotion as be
from sentiment import features as F
from sentiment import fusion
from sentiment.calibration import BaselineCalibrator
from sentiment.classifier import EmotionClassifier
from sentiment.report import Recorder
from sentiment.temporal import EmotionFilter
from vision import draw
from vision.camera import Camera
from vision.face import FaceTracker

_VIZ = list(F.LEFT_EYE) + list(F.RIGHT_EYE) + list(F.MOUTH_CORNERS) + list(F.MOUTH_VERT)
_EMO_COLORS = {
    "happy": draw.GREEN, "surprise": draw.YELLOW, "neutral": draw.DIM,
    "sad": draw.BLUE, "angry": draw.RED, "fear": (200, 120, 220),
    "disgust": (120, 200, 160),
}
_BS_EMA = 0.35  # blendshape smoothing factor (lower = steadier)


def _draw_hud(frame, smoothed, eng, feats, blink_rate, source):
    draw.panel(frame, 10, 10, 300, 250)
    draw.text(frame, "Interview analysis", 22, 36, draw.WHITE, 0.7, 2)
    draw.text(frame, f"emotion model: {source}", 22, 56, draw.DIM, 0.45)
    y = 78
    for emo in be.EMOTIONS:
        draw.text(frame, emo, 22, y + 10, draw.DIM, 0.45)
        draw.bar(frame, 110, y, 180, 12, smoothed.get(emo, 0.0),
                 _EMO_COLORS.get(emo, draw.WHITE))
        y += 22

    draw.panel(frame, frame.shape[1] - 270, 10, 260, 150)
    x0 = frame.shape[1] - 258
    draw.text(frame, "Engagement", x0, 36, draw.WHITE, 0.6, 2)
    rows = [("overall", eng.overall, draw.GREEN), ("facing", eng.facing, draw.BLUE),
            ("eye", eng.eye_contact, draw.YELLOW), ("positive", eng.positivity, draw.GREEN),
            ("composure", eng.composure, draw.BLUE)]
    yy = 52
    for name, val, col in rows:
        draw.text(frame, name, x0, yy + 9, draw.DIM, 0.4)
        draw.bar(frame, x0 + 90, yy, 150, 10, val, col)
        yy += 20

    draw.panel(frame, 10, frame.shape[0] - 70, 380, 60)
    draw.text(frame, f"yaw {feats.yaw:+5.0f}  pitch {feats.pitch:+5.0f}  roll {feats.roll:+5.0f}",
              22, frame.shape[0] - 44, draw.DIM, 0.5)
    draw.text(frame, f"blink {blink_rate:4.0f}/min   gaze ({feats.gaze_x:+.1f},{feats.gaze_y:+.1f})",
              22, frame.shape[0] - 22, draw.DIM, 0.5)


def _draw_calibration(frame, remaining):
    h, w = frame.shape[:2]
    draw.panel(frame, w // 2 - 230, h // 2 - 50, 460, 100, 0.7)
    draw.text(frame, "CALIBRATING — keep a neutral face",
              w // 2 - 210, h // 2 - 12, draw.YELLOW, 0.7, 2)
    draw.text(frame, f"starting in {remaining:.1f}s ...",
              w // 2 - 80, h // 2 + 22, draw.WHITE, 0.6)


def run(camera_index: int = 0) -> None:
    tracker = FaceTracker(num_faces=1)
    clf = EmotionClassifier.load()
    source = "SVM (trained)" if clf else "blendshape rules"
    efilter = EmotionFilter(be.EMOTIONS, stickiness=0.9)
    blink = F.BlinkCounter()
    calib = BaselineCalibrator(seconds=2.5)
    rec = Recorder()
    ema_bs: dict[str, float] = {}
    start = time.monotonic()

    print(f"Interview mode ({source}). Hold a neutral face for calibration, "
          "then react naturally. Press 'q' to finish & get the report.")

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            aspect = w / h
            res = tracker.process(frame)

            if not res.ok:
                draw.panel(frame, 10, 10, 300, 40)
                draw.text(frame, "no face detected", 22, 36, draw.RED, 0.6)
                cv2.imshow("Interview analysis  (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            draw.draw_face_points(frame, res.landmarks, _VIZ)
            feats = F.compute(res.landmarks, res.transform, aspect)

            # --- calibration phase: learn the resting baseline, no scoring yet ---
            if not calib.done:
                calib.update(res.blendshapes, feats.yaw, feats.pitch)
                rec.t0 = time.monotonic()  # don't count calibration time in the report
                start = time.monotonic()
                _draw_calibration(frame, calib.remaining())
                cv2.imshow("Interview analysis  (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # --- emotion: deviations from baseline, EMA-smoothed ---
            bs_adj = calib.adjust_blendshapes(res.blendshapes)
            for k, v in bs_adj.items():
                ema_bs[k] = _BS_EMA * v + (1 - _BS_EMA) * ema_bs.get(k, v)
            probs, valence, arousal = be.predict(ema_bs)
            if clf is not None:
                probs = clf.predict_proba(feats.feature_vector())
            smoothed = efilter.update(probs)

            if blink.update(feats.ear):
                rec.blink_total = blink.count
            blink_rate = blink.rate_per_min(time.monotonic() - start)

            # baseline-correct head pose so "facing" is relative to the sitter.
            feats_adj = replace(feats, yaw=feats.yaw - calib.base_yaw,
                                pitch=feats.pitch - calib.base_pitch)
            eng = fusion.compute(feats_adj, valence, blink_rate)
            rec.add(smoothed, eng, valence, arousal)

            top = max(smoothed, key=smoothed.get)
            draw.toast(frame, f"{top.upper()}  ({smoothed[top] * 100:.0f}%)",
                       _EMO_COLORS.get(top, draw.WHITE))
            _draw_hud(frame, smoothed, eng, feats_adj, blink_rate, source)

            cv2.imshow("Interview analysis  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    tracker.close()
    cv2.destroyAllWindows()
    rec.print_report()
    saved = rec.save()
    if saved:
        png, txt = saved
        print(f"\nGraph saved : {png}")
        print(f"Text saved  : {txt}")
