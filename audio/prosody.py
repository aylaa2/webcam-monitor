"""Voice-tone (prosody) features from raw audio — pure DSP, no ML model.

From a mono 16 kHz waveform we derive the cues a human listener uses to judge
*how* something was said (vs the words):
  - pitch (F0) via autocorrelation, plus its variation (monotone vs expressive)
  - loudness (RMS energy) and its dynamics
  - voiced ratio and pause ratio (fluency / hesitation)

These feed the HR soft-skill scoring (confidence, energy, composure).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME = 400   # 25 ms at 16 kHz
HOP = 160     # 10 ms
F0_MIN, F0_MAX = 75.0, 300.0   # human speech range (Hz)


@dataclass
class ProsodyFeatures:
    duration_s: float = 0.0
    voiced_ratio: float = 0.0
    pause_ratio: float = 0.0
    mean_f0: float = 0.0
    std_f0: float = 0.0
    f0_range: float = 0.0
    mean_energy: float = 0.0
    energy_std: float = 0.0
    available: bool = False


def _f0(frame: np.ndarray, sr: int) -> float:
    f = frame - frame.mean()
    corr = np.correlate(f, f, mode="full")[len(f) - 1:]
    if corr[0] <= 1e-9:
        return 0.0
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    seg = corr[lo:hi]
    if seg.size == 0:
        return 0.0
    lag = lo + int(np.argmax(seg))
    if corr[lag] < 0.3 * corr[0]:   # weak periodicity => unvoiced
        return 0.0
    return sr / lag


def _resample_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000 or x.size == 0:
        return x
    n = int(len(x) * 16000 / sr)
    if n < 8:
        return np.zeros(0, np.float32)
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def analyze(samples: np.ndarray, sr: int = 16000) -> ProsodyFeatures:
    x = np.asarray(samples, dtype=np.float32).flatten()
    if x.size < int(0.3 * sr):
        return ProsodyFeatures()
    # Resample to 16 kHz so the fixed FRAME/HOP sample counts match the intended
    # millisecond windows (mics often capture at 44.1/48 kHz, which otherwise
    # broke the framing -> wrong pitch and hugely inflated "pauses").
    x = _resample_16k(x, sr)
    sr = 16000
    if x.size < int(0.3 * sr):
        return ProsodyFeatures()
    x = x - x.mean()
    peak = float(np.max(np.abs(x))) + 1e-9
    x = x / peak

    n_frames = 1 + (x.size - FRAME) // HOP
    if n_frames < 4:
        return ProsodyFeatures()
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        fr = x[i * HOP: i * HOP + FRAME]
        rms[i] = float(np.sqrt(np.mean(fr * fr)))

    # speech-activity threshold (robust to dynamic range)
    thr = max(0.015, 0.10 * float(np.percentile(rms, 90)))
    active = rms > thr

    f0s = []
    for i in range(n_frames):
        if active[i]:
            v = _f0(x[i * HOP: i * HOP + FRAME], sr)
            if v > 0:
                f0s.append(v)
    f0s = np.array(f0s, dtype=np.float32)

    # Pauses: only SILENCE RUNS >= 0.25 s count (so brief inter-word/syllable gaps
    # are not mistaken for pauses), measured within the spoken span.
    idx = np.where(active)[0]
    if idx.size:
        span = active[idx[0]: idx[-1] + 1]
        min_pause = max(1, int(0.25 * sr / HOP))   # 0.25 s in frames
        pause_frames, run = 0, 0
        for v in span:
            if not v:
                run += 1
            else:
                if run >= min_pause:
                    pause_frames += run
                run = 0
        if run >= min_pause:
            pause_frames += run
        pause_ratio = pause_frames / len(span) if len(span) else 0.0
    else:
        pause_ratio = 1.0

    mean_f0 = float(f0s.mean()) if f0s.size else 0.0
    std_f0 = float(f0s.std()) if f0s.size else 0.0
    f0_range = float(np.percentile(f0s, 90) - np.percentile(f0s, 10)) if f0s.size else 0.0
    active_rms = rms[active]
    return ProsodyFeatures(
        duration_s=x.size / sr,
        voiced_ratio=float(active.mean()),
        pause_ratio=float(pause_ratio),
        mean_f0=mean_f0,
        std_f0=std_f0,
        f0_range=f0_range,
        mean_energy=float(active_rms.mean()) if active_rms.size else 0.0,
        energy_std=float(rms.std()),
        available=bool(f0s.size),
    )
