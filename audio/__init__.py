"""Audio analysis: voice-tone (prosody) DSP + offline speech-to-text (Vosk).

Both are classic, non-LLM techniques:
  - prosody features come from plain digital-signal-processing (autocorrelation
    pitch, RMS energy, pause detection)
  - transcription uses Vosk, a Kaldi/HMM-DNN recognizer that runs fully offline
"""
