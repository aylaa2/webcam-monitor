# webcam-monitor

Two real-time computer-vision systems that run **100% locally** on a laptop CPU —
**no LLM, no cloud, no internet at runtime.** Built to showcase classic,
*explainable* AI architectures.

| Feature | What it does | Core technique (non-LLM) |
|---|---|---|
| **Interview analysis** | Calibrates to your face, records you (any length), then builds a clickable multimodal **HR report** — soft + hard skills (with the exact quotes behind each), **STAR-method** coverage, background & key concepts, how you felt, voice tone + full transcript — from **face + voice + words** | Face-mesh landmarks (+ baseline calibration, speech-aware emotion) · **DSP prosody** (autocorrelation pitch, energy, pauses) · **multilingual offline ASR** (Whisper, Romanian + English) + lexicon/STAR analysis · HMM smoothing · transparent weighted fusion |
| **Gesture control** | Hand gestures trigger macOS actions (open palm → fist closes the window, ✌️ = screenshot, swipe = slides, etc.) | Hand landmarks → geometric pose rules → **finite-state machine** with a refractory lock → real OS actions |

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
- **Multimodal fusion without an LLM.** The HR report combines three independent
  signal streams — face (vision), voice tone (DSP), and words (offline ASR +
  lexicons + STAR/keyword rules) — into explainable, *evidence-linked* scores
  (click any score to see the quotes behind it). Speech is transcribed by an
  offline **ASR** model (Whisper/Vosk), and all the *analysis* on top of the
  transcript is transparent rule/lexicon logic — no language model.

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
python download_models.py          # one-time download (~2.5 GB: vision + Whisper-medium + embeddings)
```

macOS permissions:
- **Camera** — grant your terminal/IDE access (System Settings → Privacy &
  Security → Camera). Run from your own Terminal so the prompt can appear.
- **Microphone** — for the interview's voice/transcript analysis. Without it the
  interview still runs face-only and the report says so.
- **Accessibility** — needed for the keystroke gestures (close window, slides,
  play/pause, switch app, mission control). Volume, mute and screenshot use
  AppleScript / `screencapture` and work *without* it.

---

## Usage

```bash
# Feature 1 — Interview Studio (native windows, recording-style UI)
python app.py interview
python app.py interview --camera 1   # pick a different camera if needed

# Feature 2 — Gesture Control (LIVE: actually controls macOS)
python app.py gestures
python app.py gestures --dry-run    # safe preview: logs actions, controls nothing
```

### Interview Studio

A **"Interview Studio"** window with recording controls plus an **"Interview Report"**
window (HR analysis):

1. Click **Start Recording** (or press **SPACE**). Hold a **neutral face ~2.5s**
   while it calibrates to you, then speak/react naturally (the mic records too).
2. Click **Stop** / press **q** to end the recording — the **Interview Report**
   window opens with a full **HR analysis** built from **face + voice + words**:
   - **soft skills** (communication, confidence, composure, enthusiasm,
     positivity, engagement) as scored bars
   - **hard skills** — technical keywords you actually mentioned
   - **STAR method** coverage (Situation / Task / Action / Result) — detected
     **semantically** by a multilingual sentence-embedding classifier, not keywords
   - **topics discussed**, **key concepts**, and **background** highlights — the
     classifier tags each thing you said as a STAR part, background, or a
     soft/hard-skill topic (works in Romanian and English)
   - **voice & speech** metrics (pace/WPM, filler words, pitch, energy, pauses)
   - **how you felt** + **peak emotional moments** + the emotion graph
   - a written **HR recommendation** with watch-outs
3. **Click any skill, STAR, "Background", or "Full transcript"** in the report
   (look for `more >`) to open a detail window showing **the exact sentences you
   said** (with timestamps) behind that score.
4. The app stays open — **Start another recording** any time. **ESC** / **Quit**
   exits. Each session also saves a PNG + TXT in `./reports`.

> **Languages:** speech is transcribed by **Whisper (medium)** offline, which
> auto-detects the language — **Romanian (with diacritics) and English both work**
> (and ~97 others). Override the model with `WHISPER_MODEL=large-v3` (most
> accurate) or `=small` (fastest). **Recording length:** unlimited in practice
> (audio ~4 MB/min in RAM). No mic or speech model? The interview runs face-only.

### Gesture Control

A **"Gesture Control"** window (live hands + gesture legend) plus a separate
**"Action Log"** window that records every action it performs, with timestamps.

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
  camera.py  face.py  hands.py  draw.py  ui.py  textrender.py
sentiment/              Feature 1
  features.py           EAR, smile, gaze, head pose, speaking detector
  blendshape_emotion.py rule baseline (speech-aware) + valence/arousal
  calibration.py        per-person neutral baseline
  classifier.py         SVM/RandomForest wrapper
  temporal.py           HMM-style emotion filter
  fusion.py             engagement score
  report.py             emotion description + graph + window image
  hr.py                 multimodal HR report (soft/hard skills, fusion)
  run_interview.py      the recording-studio loop
  collect_data.py  train_svm.py  cnn_fer.py  train_cnn.py
audio/                  voice + words (Feature 1)
  prosody.py            DSP voice-tone features (pitch/energy/pauses)
  capture.py            background mic recording
  transcribe.py         offline ASR — Whisper (multilingual) + Vosk fallback
  lexicon.py            fillers, sentiment, action verbs, hard-skill keywords
  nlp.py                STAR cues, background + key-concept extraction (fallback)
  classify.py           multilingual embedding classifier (STAR / topic / background)
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
