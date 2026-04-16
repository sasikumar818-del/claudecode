"""Dedicated runner for the Tamil Playground 10-minute 1080p cinematic video.

Title: குழந்தைகள் விளையாட்டு பூங்காவில் மகிழ்ச்சி நாள்
       (A Joyful Day at the Playground)

FREE stack (default — no paid API keys needed):
  • Image generation : HuggingFace Inference API (FLUX.1-schnell, free)
  • Voice/TTS        : espeak-ng offline TTS (Tamil voice, no internet/key)
  • Video assembly   : MoviePy static images + audio (no Runway needed)

System pre-requisites:
  sudo apt install espeak-ng ffmpeg      # Ubuntu/Debian
  brew install espeak-ng ffmpeg          # macOS

Only ANTHROPIC_API_KEY is required (for supervisor decisions).
Optional: set HUGGINGFACE_API_KEY in .env for higher HuggingFace rate limits.

Usage:
    python videos/run_tamil_playground.py
    python videos/run_tamil_playground.py --no-supervisor   # skip Claude validation steps
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root or from videos/ subdirectory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STORYBOARD_PATH = Path(__file__).resolve().parent / "tamil_playground_storyboard.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce the Tamil Playground 10-min 1080p video (free backends)"
    )
    parser.add_argument(
        "--runway",
        action="store_true",
        help="Use Runway animation instead of static images (requires RUNWAY_API_KEY)",
    )
    parser.add_argument(
        "--tts",
        default="espeak",
        choices=["espeak", "gtts", "edge-tts", "elevenlabs"],
        help="TTS backend (default: espeak — offline, no key needed)",
    )
    parser.add_argument(
        "--dalle",
        action="store_true",
        help="Use DALL-E 3 instead of HuggingFace for images (requires OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    # ── Backend selection ──────────────────────────────────────────────
    os.environ.setdefault("IMAGE_BACKEND",  "dalle" if args.dalle else "huggingface")
    os.environ.setdefault("TTS_BACKEND",    args.tts)
    os.environ.setdefault("VIDEO_BACKEND",  "runway" if args.runway else "none")
    os.environ.setdefault("VIDEO_WIDTH",    "1920")
    os.environ.setdefault("VIDEO_HEIGHT",   "1080")

    from pipeline.orchestrator import run_pipeline

    image_label = "DALL-E 3" if args.dalle else "HuggingFace FLUX.1-schnell (free)"
    tts_label   = {
        "espeak": "espeak-ng offline (free, no internet)",
        "gtts": "Google TTS (free, needs internet)",
        "edge-tts": "Microsoft Edge TTS (free, needs internet)",
        "elevenlabs": "ElevenLabs (paid)",
    }.get(args.tts, args.tts)
    video_label = "Runway Gen-3" if args.runway else "Static images + MoviePy (free)"

    print("=" * 66)
    print("  Tamil Playground Video — Production Run")
    print("=" * 66)
    print(f"  Title    : குழந்தைகள் விளையாட்டு பூங்காவில் மகிழ்ச்சி நாள்")
    print(f"  Scenes   : 7  |  Duration : 10 minutes (600s)  |  Lang : Tamil")
    print(f"  Images   : {image_label}")
    print(f"  Voice    : {tts_label}")
    print(f"  Video    : {video_label}")
    print(f"  Output   : 1920×1080 (1080p) @ 24fps, H.264/AAC")
    print(f"  Storyboard: {STORYBOARD_PATH}")
    print("─" * 66)

    if not STORYBOARD_PATH.exists():
        print(f"\n[ERROR] Storyboard file not found: {STORYBOARD_PATH}", file=sys.stderr)
        sys.exit(1)

    package = run_pipeline(
        topic="குழந்தைகள் விளையாட்டு பூங்காவில் மகிழ்ச்சி நாள்",
        duration=600,
        tone="cinematic",
        language="ta",
        storyboard_path=STORYBOARD_PATH,
    )

    print("\n" + "=" * 66)
    print("  Production Complete!")
    print("=" * 66)
    print(f"  Preview : {package.preview_video_path}")
    print(f"  Final   : {package.final_video_path}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
