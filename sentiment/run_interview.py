"""Interview Studio — a recording-style desktop UI for facial sentiment analysis.

Two native OpenCV windows:
  - "Interview Studio"  : live camera, recording controls (clickable + keys),
                          calibration, live emotion/engagement HUD.
  - "Session Report"    : opens when you STOP a recording — your stats, a written
                          description of how you felt, peak emotional moments, and
                          the emotion graph. The app stays open so you can record
                          again; each new recording refreshes the report.

Controls:  SPACE or the on-screen button = start / stop recording
           q = stop the current recording (does NOT quit)
           ESC or the Quit button = exit

Per-recording pipeline: 2.5s neutral calibration -> baseline-relative blendshapes
(EMA smoothed) -> emotion (SVM if trained, else rules) -> HMM smoothing ->
baseline-corrected head pose -> engagement fusion -> recorded for the report.

Run:  python app.py interview  [--camera N]
"""
from __future__ import annotations

import time
from dataclasses import replace

import cv2

from sentiment import blendshape_emotion as be
from sentiment import features as F
from sentiment import emotion_model, fusion, hr, integrity
from sentiment.calibration import BaselineCalibrator
from sentiment.classifier import EmotionClassifier
from sentiment.report import Recorder
from sentiment.temporal import EmotionDecider
from audio import lexicon, prosody, transcribe
from audio.capture import AudioRecorder
from vision import draw, ui
from vision.camera import Camera
from vision.face import FaceTracker

MAIN, REPORT, DETAIL = "Interview Studio", "Interview Report", "Report - Details"
_VIZ = list(F.LEFT_EYE) + list(F.RIGHT_EYE) + list(F.MOUTH_CORNERS) + list(F.MOUTH_VERT)
_EMO_COLORS = {
    "happy": draw.GREEN, "surprise": draw.YELLOW, "neutral": draw.DIM,
    "sad": draw.BLUE, "angry": draw.RED, "fear": (200, 120, 220),
    "disgust": (120, 200, 160),
}
_BS_EMA = 0.5   # snappier blendshape smoothing so brief expressions register


def _emotion_panel(frame, smoothed):
    draw.panel(frame, 10, 90, 300, 192)
    y = 110
    for emo in be.EMOTIONS:
        draw.text(frame, emo, 22, y + 10, draw.DIM, 0.45)
        draw.bar(frame, 110, y, 180, 12, smoothed.get(emo, 0.0),
                 _EMO_COLORS.get(emo, draw.WHITE))
        y += 24


def _engagement_panel(frame, eng, feats):
    w = frame.shape[1]
    draw.panel(frame, w - 270, 10, 260, 190)
    x0 = w - 258
    draw.text(frame, "Engagement", x0, 36, draw.WHITE, 0.6, 2)
    rows = [("overall", eng.overall, draw.GREEN), ("facing", eng.facing, draw.BLUE),
            ("eye", eng.eye_contact, draw.YELLOW), ("positive", eng.positivity, draw.GREEN),
            ("composure", eng.composure, draw.BLUE)]
    yy = 52
    for name, val, col in rows:
        draw.text(frame, name, x0, yy + 9, draw.DIM, 0.4)
        draw.bar(frame, x0 + 90, yy, 150, 10, val, col)
        yy += 20
    draw.text(frame, f"yaw {feats.yaw:+4.0f} pitch {feats.pitch:+4.0f}",
              x0, yy + 8, draw.DIM, 0.4)


def _status(frame, state, elapsed, source, has_face):
    draw.panel(frame, 10, 10, 320, 72)
    draw.text(frame, "INTERVIEW STUDIO", 22, 34, draw.WHITE, 0.7, 2)
    if state == "recording":
        dot = draw.RED if int(elapsed * 2) % 2 == 0 else (60, 60, 120)
        cv2.circle(frame, (30, 56), 7, dot, -1, cv2.LINE_AA)
        draw.text(frame, f"REC  {int(elapsed // 60)}:{int(elapsed % 60):02d}",
                  46, 62, draw.RED, 0.6, 2)
    elif state == "calibrating":
        draw.text(frame, "CALIBRATING", 22, 62, draw.YELLOW, 0.6, 2)
    else:
        draw.text(frame, "READY", 22, 62, draw.GREEN, 0.6, 2)
    draw.text(frame, source, 175, 62, draw.DIM, 0.4)
    if not has_face and state != "idle":
        draw.text(frame, "show your face", 175, 34, draw.RED, 0.45)


def _calibration_overlay(frame, remaining):
    h, w = frame.shape[:2]
    draw.panel(frame, w // 2 - 230, h // 2 - 50, 460, 100, 0.7)
    draw.text(frame, "CALIBRATING — keep a neutral face",
              w // 2 - 210, h // 2 - 12, draw.YELLOW, 0.7, 2)
    draw.text(frame, f"starting in {remaining:.1f}s ...",
              w // 2 - 80, h // 2 + 22, draw.WHITE, 0.6)


def run(camera_index: int = 0, lang: str | None = None) -> None:
    tracker = FaceTracker(num_faces=2)   # 2 so a second person can be flagged
    clf = EmotionClassifier.load()
    use_hse = emotion_model.available()
    source = ("HSEmotion + rules" if use_hse else
              "SVM (trained)" if clf else "blendshape rules")

    cv2.namedWindow(MAIN)
    mouse = ui.MouseState()
    cv2.setMouseCallback(MAIN, mouse)
    cv2.moveWindow(MAIN, 60, 60)

    state = "idle"
    rec = decider = blink = calib = speak = audio = hse = None
    ema_bs: dict[str, float] = {}
    rec_start = 0.0
    second_face_seen = False
    lang_choice = lang if lang in ("en", "ro") else "auto"
    last_smoothed: dict[str, float] = {}
    report_placed = detail_placed = False
    report_mouse = ui.MouseState()
    last_hr = None
    report_buttons: list = []
    report_scale = 1.0

    def begin():
        nonlocal rec, decider, blink, calib, speak, audio, hse, ema_bs, state
        nonlocal second_face_seen
        rec = Recorder()
        decider = EmotionDecider(be.EMOTIONS)  # stable, committed emotion decision
        blink = F.BlinkCounter()
        calib = BaselineCalibrator(seconds=2.5)
        speak = F.SpeakingDetector()
        audio = AudioRecorder()
        hse = emotion_model.Throttle(every=3)
        ema_bs = {}
        second_face_seen = False
        state = "calibrating"

    def finish():
        nonlocal state, report_placed, last_hr, report_buttons, report_scale
        was = state
        state = "idle"
        if was != "recording" or not rec or not rec.t:
            if audio:
                audio.stop()
            return

        audio_res = audio.stop() if audio else None
        face = rec.summary()
        png, _txt = rec.save()
        rec.print_report()

        if audio_res and audio_res.available:
            print(f"Transcribing (Whisper, lang={lang_choice})...")
            forced = None if lang_choice == "auto" else lang_choice
            tr = transcribe.transcribe(audio_res.samples, audio_res.sr, language=forced)
            pf = prosody.analyze(audio_res.samples, audio_res.sr)
        else:
            tr, pf = transcribe.Transcript(), prosody.ProsodyFeatures()
        ws = lexicon.analyze(tr.text)
        hrr = hr.compute(face, pf, ws, face["duration_s"], transcript=tr.text,
                         segments=tr.segments, language=tr.language, engine=tr.engine)
        hrr.emotion_description = rec.describe()
        hrr.emotion_moments = hr.attach_quotes(rec.emotional_moments(), tr.segments)
        hrr.integrity = integrity.compute(rec, pf, ws, second_face_seen)
        print("\nHR SUMMARY:", hrr.narrative)
        last_hr = hrr

        img, buttons = hr.render_image(hrr, face, png)
        sc = min(1.0, 1040 / img.shape[0])
        report_scale = sc
        report_buttons = buttons
        if sc < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
        cv2.namedWindow(REPORT)
        cv2.setMouseCallback(REPORT, report_mouse)
        cv2.imshow(REPORT, img)
        if not report_placed:
            cv2.moveWindow(REPORT, 90, 90)
            report_placed = True

    def open_detail(key):
        nonlocal detail_placed
        img = hr.render_detail(last_hr, key)
        sc = min(1.0, 980 / img.shape[0])
        if sc < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
        cv2.imshow(DETAIL, img)
        if not detail_placed:
            cv2.moveWindow(DETAIL, 150, 150)
            detail_placed = True

    print("Interview Studio open. SPACE/Start to record, q to stop, ESC to quit.")

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            res = tracker.process(frame)
            has_face = res.ok
            elapsed = (time.monotonic() - rec_start) if state == "recording" else 0.0

            if has_face and state != "idle":
                draw.draw_face_points(frame, res.landmarks, _VIZ)
                feats = F.compute(res.landmarks, res.transform, w / h)

                if state == "calibrating":
                    calib.update(res.blendshapes, feats.yaw, feats.pitch,
                                 feats.gaze_x, feats.gaze_y)
                    _calibration_overlay(frame, calib.remaining())
                    if calib.done:
                        state = "recording"
                        rec.t0 = rec_start = time.monotonic()
                        audio.start()   # begin mic capture once calibration is done
                elif state == "recording":
                    if res.n_faces > 1:
                        second_face_seen = True
                    speaking = speak.update(res.blendshapes.get("jawOpen", 0.0))
                    bs_adj = calib.adjust_blendshapes(res.blendshapes)
                    for k, v in bs_adj.items():
                        ema_bs[k] = _BS_EMA * v + (1 - _BS_EMA) * ema_bs.get(k, v)
                    rule_probs, valence, arousal = be.predict(ema_bs, speaking=speaking)
                    model_probs = hse.step(frame, res.landmarks)
                    if model_probs is not None:
                        probs = emotion_model.fuse(model_probs, rule_probs, speaking=speaking)
                    elif clf is not None:
                        probs = clf.predict_proba(feats.feature_vector())
                    else:
                        probs = rule_probs
                    committed, smoothed = decider.update(probs)
                    last_smoothed = smoothed
                    if blink.update(feats.ear):
                        rec.blink_total = blink.count
                    blink_rate = blink.rate_per_min(time.monotonic() - rec_start)
                    feats_adj = replace(feats, yaw=feats.yaw - calib.base_yaw,
                                        pitch=feats.pitch - calib.base_pitch,
                                        gaze_x=feats.gaze_x - calib.base_gaze_x,
                                        gaze_y=feats.gaze_y - calib.base_gaze_y)
                    eng = fusion.compute(feats_adj, valence, blink_rate)
                    rec.add(smoothed, eng, valence, arousal,
                            gaze_x=feats_adj.gaze_x, gaze_y=feats_adj.gaze_y,
                            yaw=feats_adj.yaw, pitch=feats_adj.pitch)

                    top = committed
                    draw.toast(frame, f"{top.upper()}  ({smoothed[top] * 100:.0f}%)",
                               _EMO_COLORS.get(top, draw.WHITE))
                    _emotion_panel(frame, smoothed)
                    _engagement_panel(frame, eng, feats_adj)

            _status(frame, state, elapsed, source, has_face)

            # --- control bar ---
            draw.panel(frame, 0, h - 64, w, 64, 0.5)
            if state == "recording":
                primary = ui.Button(w // 2 - 210, h - 52, 200, 40, "Stop Recording", draw.RED)
            elif state == "calibrating":
                primary = ui.Button(w // 2 - 210, h - 52, 200, 40, "Calibrating...",
                                    (70, 70, 70))
                primary.enabled = False
            else:
                primary = ui.Button(w // 2 - 210, h - 52, 200, 40, "Start Recording",
                                    (80, 150, 80))
            quit_btn = ui.Button(w // 2 + 20, h - 52, 120, 40, "Quit", (90, 90, 90))
            primary.draw(frame, mouse.hover)
            quit_btn.draw(frame, mouse.hover)
            lang_name = {"auto": "AUTO", "en": "English", "ro": "Romana"}[lang_choice]
            draw.text(frame, f"lang: {lang_name}  ('l' to change)", 22, h - 24, draw.YELLOW, 0.5)
            draw.text(frame, "SPACE start/stop  -  q stop  -  ESC quit",
                      w - 360, h - 24, draw.DIM, 0.45)

            click = mouse.take_click()
            if quit_btn.contains(click):
                break
            if primary.enabled and primary.contains(click):
                begin() if state == "idle" else finish()

            cv2.imshow(MAIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                      # ESC
                break
            elif key == 32:                    # SPACE
                if state == "idle":
                    begin()
                elif state == "recording":
                    finish()
            elif key == ord("q") and state == "recording":
                finish()
            elif key == ord("l"):              # cycle transcript language
                lang_choice = {"auto": "en", "en": "ro", "ro": "auto"}[lang_choice]

            # report window: click a skill / section for a "what was said" detail
            rc = report_mouse.take_click()
            if rc and last_hr and report_buttons:
                ox, oy = rc[0] / report_scale, rc[1] / report_scale
                for x1, y1, x2, y2, bkey in report_buttons:
                    if x1 <= ox <= x2 and y1 <= oy <= y2:
                        open_detail(bkey)
                        break

    if audio:
        audio.stop()   # release the mic if we quit mid-recording
    tracker.close()
    cv2.destroyAllWindows()
