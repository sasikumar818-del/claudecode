#!/usr/bin/env python3
"""Runner: குழந்தைகள் விளையாட்டு பூங்காவில் மகிழ்ச்சி நாள்

Zero-cost 10-minute Tamil children's playground video.

Usage:
  # Full free run (espeak TTS + HuggingFace images):
  python videos/run_tamil_playground.py

  # Offline only (espeak + PIL gradient images, no internet needed):
  python videos/run_tamil_playground.py --offline

  # With HuggingFace token for better image quota:
  python videos/run_tamil_playground.py --hf-token hf_xxxxx

  # Use gTTS for clearer Tamil voice (needs internet):
  python videos/run_tamil_playground.py --tts gtts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce the Tamil playground video (zero-cost pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: use PIL images + espeak TTS (no internet required)",
    )
    parser.add_argument(
        "--tts",
        default="espeak",
        choices=["espeak", "gtts", "edge-tts"],
        help="TTS backend (default: espeak)",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        metavar="TOKEN",
        help="HuggingFace API token for higher image quota",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root output directory (default: output)",
    )
    args = parser.parse_args()

    storyboard = Path(__file__).parent / "tamil_playground_storyboard.json"
    if not storyboard.exists():
        print(f"ERROR: storyboard not found at {storyboard}", file=sys.stderr)
        sys.exit(1)

    image_backend = "pil" if args.offline else "huggingface"
    tts_backend   = "espeak" if args.offline else args.tts

    from pipeline.free_pipeline import run_free_pipeline

    package = run_free_pipeline(
        storyboard_path=str(storyboard),
        language="ta",
        image_backend=image_backend,
        tts_backend=tts_backend,
        output_dir=args.output_dir,
        video_width=1920,
        video_height=1080,
        hf_api_key=args.hf_token,
    )

    print(f"\nDone! Your video is ready:")
    print(f"  {package.final_video_path}\n")


if __name__ == "__main__":
    main()
