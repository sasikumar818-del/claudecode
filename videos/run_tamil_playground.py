"""Dedicated runner for the Tamil Playground 10-minute 1080p cinematic video.

Title: குழந்தைகள் விளையாட்டு பூங்காவில் மகிழ்ச்சி நாள்
       (A Joyful Day at the Playground)

Usage:
    python videos/run_tamil_playground.py [--no-animate]

Prerequisites (.env):
    ANTHROPIC_API_KEY=...
    ELEVENLABS_API_KEY=...
    ELEVENLABS_VOICE_ID_TA=<Tamil ElevenLabs voice ID>   # find at elevenlabs.io/voice-library
    OPENAI_API_KEY=...
    RUNWAY_API_KEY=...         # optional; omit or use --no-animate for static images
    VIDEO_WIDTH=1920
    VIDEO_HEIGHT=1080
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
        description="Produce the Tamil Playground 10-min 1080p Pixar-style video"
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Use static images instead of Runway animation (faster, cheaper)",
    )
    args = parser.parse_args()

    if args.no_animate:
        os.environ["VIDEO_BACKEND"] = "none"

    # Force 1080p if not already set in environment
    os.environ.setdefault("VIDEO_WIDTH", "1920")
    os.environ.setdefault("VIDEO_HEIGHT", "1080")

    from pipeline.orchestrator import run_pipeline

    print("=" * 62)
    print("  Tamil Playground Video — Production Run")
    print("=" * 62)
    print(f"  Title    : குழந்தைகள் விளையாட்டு பூங்காவில் மகிழ்ச்சி நாள்")
    print(f"  Scenes   : 7")
    print(f"  Duration : 10 minutes (600 seconds)")
    print(f"  Language : Tamil (ta)")
    print(f"  Style    : Pixar cinematic, 1920×1080")
    print(f"  Animate  : {'No (static images)' if args.no_animate else 'Yes (Runway Gen-3)'}")
    print(f"  Storyboard: {STORYBOARD_PATH}")
    print("─" * 62)

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

    print("\n" + "=" * 62)
    print("  Production Complete!")
    print("=" * 62)
    print(f"  Preview : {package.preview_video_path}")
    print(f"  Final   : {package.final_video_path}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
