"""Offline speech-to-text with timestamped segments.

Primary engine: faster-whisper (multilingual incl. Romanian, auto language
detection, accurate). Fallback: Vosk (English, lighter). Both run fully offline
once their model is downloaded. Returns sentence-level segments with timestamps
so the HR report can quote *exactly what was said* and when.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parent.parent / "models"
WHISPER_DIR = _MODELS / "whisper"
VOSK_DIR = _MODELS / "vosk-model-small-en-us-0.15"
# Override with e.g. WHISPER_MODEL=large-v3 (most accurate) or =small (fastest).
WHISPER_SIZE = os.environ.get("WHISPER_MODEL", "medium")
SR = 16000

_whisper_model = None  # cached across recordings


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str = ""
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    engine: str = "none"
    available: bool = False


def _resample_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).flatten()
    if sr == SR or x.size == 0:
        return x
    n = int(len(x) * SR / sr)
    if n <= 0:
        return np.zeros(0, np.float32)
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def transcribe(samples: np.ndarray, sr: int) -> Transcript:
    audio = _resample_16k(samples, sr)
    if audio.size < SR // 2:           # < 0.5 s of audio
        return Transcript()
    return _whisper(audio) or _vosk(audio) or Transcript()


def _whisper(audio: np.ndarray) -> Transcript | None:
    try:
        from faster_whisper import WhisperModel
    except Exception:  # noqa: BLE001
        return None
    global _whisper_model
    try:
        if _whisper_model is None:
            _whisper_model = WhisperModel(WHISPER_SIZE, device="cpu",
                                         compute_type="int8",
                                         download_root=str(WHISPER_DIR))
        seg_gen, info = _whisper_model.transcribe(audio, beam_size=5, vad_filter=True)
        segs, texts = [], []
        for s in seg_gen:
            txt = s.text.strip()
            if txt:
                segs.append(Segment(float(s.start), float(s.end), txt))
                texts.append(txt)
        full = " ".join(texts).strip()
        lang = getattr(info, "language", "") or ""
        return Transcript(full, segs, lang, "whisper", bool(full))
    except Exception as exc:  # noqa: BLE001
        print(f"[audio] whisper transcription failed: {exc}")
        return None


def _vosk(audio: np.ndarray) -> Transcript | None:
    if not VOSK_DIR.exists():
        return None
    try:
        import json

        import vosk
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(VOSK_DIR))
        rec = vosk.KaldiRecognizer(model, SR)
        rec.SetWords(True)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        segs, texts = [], []

        def add(js: str) -> None:
            r = json.loads(js)
            t = r.get("text", "").strip()
            if not t:
                return
            ws = r.get("result", [])
            st = ws[0]["start"] if ws else 0.0
            en = ws[-1]["end"] if ws else 0.0
            segs.append(Segment(float(st), float(en), t))
            texts.append(t)

        step = 8000
        for i in range(0, len(pcm), step):
            if rec.AcceptWaveform(pcm[i:i + step]):
                add(rec.Result())
        add(rec.FinalResult())
        full = " ".join(texts).strip()
        return Transcript(full, segs, "en", "vosk", bool(full))
    except Exception as exc:  # noqa: BLE001
        print(f"[audio] vosk transcription failed: {exc}")
        return None
