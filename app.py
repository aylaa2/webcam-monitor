"""webcam-monitor — two real-time, fully-local computer-vision demos.

    python app.py interview            # facial sentiment & engagement analysis
    python app.py gestures             # hand-gesture control (LIVE, controls macOS)
    python app.py gestures --dry-run   # ... safe preview, prints actions only

Both run 100% offline. First-time setup:
    python -m pip install -r requirements.txt
    python download_models.py
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_int = sub.add_parser("interview", help="facial sentiment & engagement analysis")
    p_int.add_argument("--camera", type=int, default=0)
    p_int.add_argument("--lang", choices=["auto", "en", "ro"], default=None,
                       help="transcript language (default: auto; toggle live with 'l')")

    p_ges = sub.add_parser("gestures", help="hand-gesture control")
    p_ges.add_argument("--camera", type=int, default=0)
    p_ges.add_argument("--dry-run", action="store_true",
                       help="safe preview: print actions without controlling the OS")

    args = parser.parse_args()

    if args.mode == "interview":
        from sentiment.run_interview import run
        run(camera_index=args.camera, lang=args.lang)
    elif args.mode == "gestures":
        from gestures.run_gestures import run
        run(camera_index=args.camera, live=not args.dry_run)


if __name__ == "__main__":
    main()
