# webcam-monitor

Two real-time computer-vision systems that run **100% locally** on a laptop CPU —
**no LLM, no cloud, no internet at runtime.** Built to showcase classic,
*explainable* AI architectures.

| Feature | What it does | Core technique (non-LLM) |
|---|---|---|
| **Interview analysis** | Calibrates to your neutral face, reads live sentiment + engagement, then writes a description of how you felt with the timestamps of your most emotional moments + an annotated graph | Face-mesh landmarks → **per-person baseline calibration** → hand-crafted geometric features → SVM / RandomForest + rule baseline → **HMM temporal smoothing** → weighted multi-cue fusion |
| **Gesture control** | Hand gestures trigger macOS actions (open palm → fist closes the window, swipe to change slides, etc.) | Hand landmarks → geometric pose rules → **finite-state machine** for dynamic gestures → keystroke actions |

Everything below the landmark detector is transparent math you can point at and
explain — which is the whole pedagogical point.

---

## Why this is interesting from a (non-LLM) architecture perspective

- **Edge AI / privacy-preserving.** All inference is on-device; no frame ever
  leaves the machine.
- **Two complementary paradigms in one repo.** A *static* probabilistic
  classifier (sentiment) and a *temporal* deterministic recognizer (gestures).
- **Explainable by construction.** Every decision traces back to specific
  landmarks, angles, and weights — the opposite of a black box.
- **A built-in experiment.** The same emotion task is solved three ways with
  increasing model complexity, so you can compare accuracy *and* interpretability:
  1. **Rule baseline** — weighted sums of 52 facial *blendshapes* (zero training).
  2. **Classic ML** — geometric features + **SVM (RBF)** vs **RandomForest**.
  3. **Deep learning** — a small **CNN** trained on FER2013 (optional).
- **Classic probabilistic + automata theory.** A **Hidden-Markov-style filter**
  steadies the emotion signal; a **finite-state machine** parses gesture grammar.

---

## Architecture

```
                         ┌──────────────────────┐
            Webcam ─────▶│  OpenCV capture       │  (threaded, ~30 fps)
                         └──────────┬───────────┘
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
   FACE pipeline (sentiment)                     HAND pipeline (gesture)
   MediaPipe Face Landmarker                     MediaPipe Hand Landmarker
   478 pts + 52 blendshapes + pose               21 keypoints/hand
              │                                             │
   feature extraction                            static pose (finger rules)
   EAR · smile · gaze · head pose                          │
              │                                   gesture state machine
   emotion: rules | SVM | CNN                     open→fist · swipe · pinch
              │                                             │
   HMM temporal smoothing                          action controller
              │                                     pyautogui hotkeys
   multi-cue engagement fusion
              │
   live HUD + end-of-session report
```

---

## Setup (macOS, Python 3.12)

> MediaPipe currently has no wheels for Python 3.13/3.14 — use 3.12.

```bash
# from the project root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py          # one-time ~10 MB model download
```

macOS permissions:
- **Camera** — grant your terminal/IDE access (System Settings → Privacy &
  Security → Camera). Run from your own Terminal so the prompt can appear.
- **Accessibility** — needed for the keystroke gestures (close window, slides,
  play/pause, switch app, mission control). Volume, mute and screenshot use
  AppleScript / `screencapture` and work *without* it.

---

## Usage

```bash
# Feature 1 — interview sentiment analysis
python app.py interview
#   1) hold a NEUTRAL face ~2.5s while it calibrates to you
#   2) react naturally; press 'q' to finish
#   -> prints a written description of how you felt + peak emotional moments,
#      and saves an annotated graph + text report to ./reports

# Feature 2 — gesture control (LIVE: actually controls macOS)
python app.py gestures
python app.py gestures --dry-run   # safe preview: prints actions, controls nothing
```

### Gesture cheat-sheet

| Gesture | Action |
|---|---|
| ✊ open palm → close to a fist | Close window (Cmd+W) |
| 👉 / ✋ swipe right / left | Next / previous slide (→ / ←) |
| ✋ swipe up / down | Volume up / down |
| 🤏 pinch (thumb+index) | Mission Control |
| 👍 thumbs up | Play / Pause |
| 👎 thumbs down | Mute |
| ✌️ peace sign (hold) | Screenshot |
| 🙌 two open palms | Switch app (Cmd+Tab) |

A per-action **cooldown** and a **hold-to-confirm** on static gestures prevent
accidental triggers.

---

## Training the classic model on your own data (no download needed)

```bash
python -m sentiment.collect_data     # press 0-6 to label faces from the webcam
python -m sentiment.train_svm        # SVM vs RandomForest, CV + confusion matrix
python app.py interview              # now uses your trained SVM automatically
```

`train_svm.py` prints **RandomForest feature importances** — e.g. how much
`smile`, `brow_raise`, or `gaze` each contribute — a ready-made interpretability
slide.

## Optional: the CNN comparison

```bash
pip install -r requirements-ml.txt
# download fer2013.csv into ./data, then:
python -m sentiment.train_cnn --epochs 30
```

Quote the CNN's accuracy next to the SVM's for a complete "hand-crafted features
vs learned features" experiments section.

---

## Project structure

```
app.py                  entry point (interview | gestures)
download_models.py      fetch MediaPipe .task bundles
vision/                 shared core: camera, face/hand trackers, drawing
  camera.py  face.py  hands.py  draw.py
sentiment/              Feature 1
  features.py           EAR, smile, gaze, head-pose geometry
  blendshape_emotion.py rule baseline (+ valence/arousal)
  classifier.py         SVM/RandomForest wrapper
  temporal.py           HMM-style emotion filter
  fusion.py             engagement score
  report.py             debrief + timeline plot
  run_interview.py      the live loop
  collect_data.py  train_svm.py  cnn_fer.py  train_cnn.py
gestures/               Feature 2
  poses.py              static pose geometry
  state_machine.py      dynamic gesture FSM
  actions.py            macOS action mapping
  run_gestures.py       the live loop
```

## Limitations / honest notes

- Head-pose Euler angles and gaze are approximate; signs may need calibrating to
  your camera. They're indicative, not clinical.
- The rule/SVM emotion read is a coarse 7-class estimate — affective computing is
  hard, and that's a fair thing to say in the presentation.
- Gesture thresholds (swipe distance, cooldowns) are tuned for a typical
  arm's-length webcam setup; adjust in `state_machine.py` if needed.
