# webcam-monitor

Two real-time computer-vision systems that run **100% locally** on a laptop CPU —
**no LLM, no cloud, no internet at runtime.** Built to showcase classic,
*explainable* AI architectures.

| Feature | What it does | Core technique (non-LLM) |
|---|---|---|
| **Interview analysis** | Calibrates to your face, records you (any length), then builds a clickable multimodal **HR report** — every **hard + soft skill you mentioned** (with the exact quotes), background, key concepts, how you felt, voice tone + full transcript — from **face + voice + words** | **AffectNet face-emotion model (HSEmotion) ensembled with blendshape rules** (+ calibration, speech-aware) · **DSP prosody** · **multilingual offline ASR** (Whisper) · **multilingual embedding classifier** for skills/background + **semantic keyphrases (MMR)** · HMM smoothing · transparent fusion |
| **Gesture control** | Hand gestures drive macOS — **finger-controlled cursor with dwell-click**, open palm→fist closes the window, finger-circle = volume, ✌️ = screenshot, etc. | Hand landmarks → wrist-relative pose rules → **finite-state machine** (refractory lock) + dwell-click controller → real OS actions |

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
- **A built-in experiment.** The same emotion task is solved several ways with
  increasing model complexity, so you can compare accuracy *and* interpretability:
  1. **Rule baseline** — weighted sums of 52 facial *blendshapes* (zero training).
  2. **Classic ML** — geometric features + **SVM (RBF)** vs **RandomForest**.
  3. **Deep learning** — a small **CNN** trained on FER2013 (optional), or the
     bundled **AffectNet model (HSEmotion)** which is the default, **ensembled**
     (late-fusion) with the rules.
- **Measurable accuracy + an honest negative result.** Trainers print cross-val
  accuracy + confusion matrices; and a tried-but-rejected cross-encoder reranker
  (kept behind `CLASSIFY_RERANK=1`) that did *not* beat the bi-encoder on this
  task — a real empirical comparison rather than hand-waving.
- **Classic probabilistic + automata theory.** A **Hidden-Markov-style filter**
  steadies the emotion signal; a **finite-state machine** parses gesture grammar.
- **Multimodal fusion without an LLM.** The HR report combines three independent
  signal streams — face (vision), voice tone (DSP), and words (offline ASR +
  lexicons + keyword rules) — into explainable, *evidence-linked* scores
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
   - **soft-skill scores** (communication, confidence, composure, enthusiasm,
     positivity, engagement) as bars
   - **all hard + soft skills you mentioned** — detected **semantically** by a
     multilingual embedding classifier (works in Romanian + English) *and* a
     keyword backup, so nothing is missed
   - **background** highlights + **key concepts** (semantic keyphrases)
   - **voice & speech** metrics (pace/WPM, filler words, pitch, energy, pauses)
   - **how you felt** + **peak emotional moments** (with **the exact line you said**
     at each peak) + the emotion graph
   - an **interview-integrity** heuristic — flags signals like frequent look-away,
     consistent off-screen gaze, scripted delivery, or a second face (clearly
     labelled *heuristic, not proof of cheating*)
   - a written **HR recommendation** with watch-outs
3. **Click any skill, "Background", or "Full transcript"** in the report
   (look for `more >`) to open a detail window showing **the exact sentences you
   said** (with timestamps) behind it.
4. The app stays open — **Start another recording** any time. **ESC** / **Quit**
   exits. Each session also saves a PNG + TXT in `./reports`.

> **Languages:** speech is transcribed by **Whisper (medium)** offline. Auto-detect
> is **constrained to English + Romanian** (so it won't drift to other languages),
> and you can lock it: press **`l`** in the window to cycle AUTO / English / Romanian,
> pass **`--lang en`**, or set `INTERVIEW_LANG=en`. Override the model with
> `WHISPER_MODEL=large-v3` (most accurate) or `=small` (fastest). **Recording
> length:** unlimited in practice. No mic or speech model? The interview runs face-only.

### Gesture Control

A **"Gesture Control"** window (live hands + gesture legend) plus a separate
**"Action Log"** window that records every action it performs, with timestamps.

### Gesture cheat-sheet

| Gesture | Action |
|---|---|
| ☝️ point your index finger | Move the mouse cursor |
| ☝️⏸ hold the cursor still ~2.5 s (after moving it) | Left click |
| ✊ open palm → close to a fist | Close window (Cmd+W) |
| ☝️🔄 draw a circle with your index finger | Volume up (clockwise) / down (counter-clockwise) |
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

### Train the learned hand-pose classifier (replaces the geometric thresholds)

```bash
python -m gestures.collect_data      # press 1-7 to label hand poses from the webcam
python -m gestures.train_pose        # RandomForest, CV accuracy + confusion matrix
python app.py gestures               # poses.classify now uses your trained model
```

Without a trained model the gestures use the geometric rules (unchanged); once
trained, `poses.classify` switches to the learned classifier with a confidence
gate and falls back to geometry when unsure.

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
  emotion_model.py      AffectNet face model (HSEmotion) + late-fusion ensemble
  calibration.py        per-person neutral baseline
  classifier.py         SVM/RandomForest wrapper
  temporal.py           HMM-style emotion filter
  fusion.py             engagement score (head + gaze eye-contact)
  report.py             emotion description + graph + window image
  integrity.py          heuristic interview-integrity signals (gaze/delivery/2nd face)
  hr.py                 multimodal HR report (soft/hard skills, fusion)
  run_interview.py      the recording-studio loop
  collect_data.py  train_svm.py  cnn_fer.py  train_cnn.py
audio/                  voice + words (Feature 1)
  prosody.py            DSP voice-tone features (pitch/energy/pauses)
  capture.py            background mic recording
  transcribe.py         offline ASR — Whisper (multilingual) + Vosk fallback
  lexicon.py            fillers, sentiment, action verbs, hard-skill keywords
  nlp.py                background + key-concept extraction (lexicon fallback)
  classify.py           multilingual classifier (hard/soft skills, background) + keyphrases
gestures/               Feature 2
  poses.py              static pose: learned classifier (if trained) or geometry
  state_machine.py      dynamic gesture FSM (grab, finger-circle volume, etc.)
  cursor.py             finger-controlled mouse + dwell-to-click
  actions.py            macOS action mapping
  collect_data.py       record labeled pose samples
  train_pose.py         train + evaluate the learned pose classifier
  run_gestures.py       the live loop
```

## Limitations / honest notes

- Head-pose Euler angles and gaze are approximate; signs may need calibrating to
  your camera. They're indicative, not clinical.
- The rule/SVM emotion read is a coarse 7-class estimate — affective computing is
  hard, and that's a fair thing to say in the presentation.
- Gesture thresholds (swipe distance, cooldowns) are tuned for a typical
  arm's-length webcam setup; adjust in `state_machine.py` if needed.
