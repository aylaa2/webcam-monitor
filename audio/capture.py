"""Background microphone recording (sounddevice).

Records the interview audio on a background thread and returns the raw waveform.
Transcription + prosody happen afterwards (audio/transcribe.py, audio/prosody.py).
Guarded: if sounddevice or a mic is missing, the interview runs face-only.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

SR = 16000

try:
    import sounddevice as sd
    _HAVE_SD = True
except Exception:  # noqa: BLE001
    _HAVE_SD = False


@dataclass
class AudioResult:
    samples: np.ndarray
    sr: int
    available: bool


class AudioRecorder:
    def __init__(self) -> None:
        self.available = _HAVE_SD
        self.sr = SR
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.available:
            return
        self._chunks = []
        last = None
        for rate in (SR, None):   # prefer 16 kHz, else device default
            try:
                kw = {} if rate is None else {"samplerate": rate}
                self._stream = sd.InputStream(channels=1, dtype="float32",
                                              callback=self._cb, **kw)
                self._stream.start()
                self.sr = int(self._stream.samplerate)
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
        print(f"[audio] microphone unavailable: {last}")
        self.available = False

    def _cb(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        with self._lock:
            self._chunks.append(indata.copy().reshape(-1))

    def stop(self) -> AudioResult:
        if not self.available or self._stream is None:
            return AudioResult(np.zeros(0, np.float32), self.sr, False)
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass
        self._stream = None
        with self._lock:
            samples = (np.concatenate(self._chunks) if self._chunks
                       else np.zeros(0, np.float32))
        return AudioResult(samples, self.sr, samples.size > 0)
