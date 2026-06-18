import threading
import librosa
import numpy as np
import sounddevice as sd

class VoiceAnalyzer:
    def __init__(self, duration=3.0, sr=22050):
        self.duration = duration
        self.sr = sr
        self.current_confidence = 100
        self.is_running = False
        self._thread = None

    def start(self):
        self.is_running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join()

    def _listen_loop(self):
        while self.is_running:
            try:
                # Capture a short chunk of audio
                audio_data = sd.rec(int(self.duration * self.sr), samplerate=self.sr, channels=1, dtype='float32')
                sd.wait()
                audio = audio_data.flatten()

                # Calculate Pitch Variance (F0)
                f0, voiced_flag, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                valid_f0 = f0[voiced_flag] if f0 is not None else []
                
                pitch_variance = np.std(valid_f0) if len(valid_f0) > 0 else 0

                # Calculate Hesitation (Silences)
                rms_energy = librosa.feature.rms(y=audio)[0]
                silence_frames = np.sum(rms_energy < 0.01)
                hesitation_ratio = silence_frames / len(rms_energy) if len(rms_energy) > 0 else 1

                # Basic Ruleset for Confidence
                score = 100
                if pitch_variance < 10.0:
                    score -= 30  # Monotone penalty
                if hesitation_ratio > 0.3:
                    score -= 40  # Hesitation penalty

                self.current_confidence = max(0, score)
            except Exception as e:
                # Silently catch mic errors so it doesn't crash the video
                pass