"""Download the MediaPipe Tasks model bundles into ./models.

These are small (~3-8 MB each) and are fetched ONCE. After this, the whole
project runs fully offline — nothing leaves your machine at runtime.

Usage:
    python download_models.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

MODELS = {
    # Face mesh + 52 blendshapes + 4x4 facial transformation matrix (head pose).
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    # 21 hand keypoints + handedness.
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

# Offline speech-to-text model (Vosk) for the HR interview report. Unzips to a
# folder; skipped automatically if already present.
VOSK_ZIP_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_DIR_NAME = "vosk-model-small-en-us-0.15"

MODELS_DIR = Path(__file__).parent / "models"


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, blocks * block_size * 100 // total)
    sys.stdout.write(f"\r    {pct:3d}%")
    sys.stdout.flush()


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}")
        try:
            urllib.request.urlretrieve(url, dest, _progress)
            print(f"\r    done -> {dest}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[fail] could not download {name}: {exc}")
            return 1

    # Vosk speech model (zip -> folder).
    vosk_dir = MODELS_DIR / VOSK_DIR_NAME
    if vosk_dir.exists():
        print(f"[skip] {VOSK_DIR_NAME} already present")
    else:
        print(f"[get ] {VOSK_DIR_NAME} (offline speech-to-text)")
        try:
            import zipfile

            zip_path = MODELS_DIR / "vosk.zip"
            urllib.request.urlretrieve(VOSK_ZIP_URL, zip_path, _progress)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(MODELS_DIR)
            zip_path.unlink()
            print(f"\r    done -> {vosk_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[warn] speech model failed ({exc}); interview will run face-only.")

    # Whisper (faster-whisper) — multilingual ASR for the HR report.
    whisper_size = os.environ.get("WHISPER_MODEL", "medium")
    whisper_dir = MODELS_DIR / "whisper"
    print(f"[get ] whisper {whisper_size} (multilingual speech-to-text)")
    try:
        from faster_whisper import WhisperModel

        WhisperModel(whisper_size, device="cpu", compute_type="int8",
                     download_root=str(whisper_dir))
        print(f"    done -> {whisper_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] whisper download skipped ({exc}); will fall back to Vosk/face-only.")

    # AffectNet face-emotion model (HSEmotion, ONNX) — ensembled with the rules.
    print("[get ] HSEmotion face-emotion model (AffectNet)")
    try:
        import urllib.request as _r  # noqa: F401  (ensures hsemotion's download works)
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
        print("    done")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] HSEmotion skipped ({exc}); emotion will use rules only.")

    # Multilingual sentence-embedding model for the semantic STAR/topic classifier.
    print("[get ] embedding model (semantic STAR / topic classifier)")
    try:
        from fastembed import TextEmbedding

        TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                      cache_dir=str(MODELS_DIR / "embed"))
        print(f"    done -> {MODELS_DIR / 'embed'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] embedding model skipped ({exc}); will fall back to keyword rules.")

    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
