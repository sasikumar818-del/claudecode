"""Main entry point — CLI for the AI Video Production Pipeline."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Video Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "The history of space exploration" --duration 90 --tone cinematic
  python main.py --topic "How black holes form" --duration 60 --no-animate
  python main.py --topic "Top 5 Python tips" --tone educational
        """,
    )
    parser.add_argument("--topic", required=True, help="Video topic or prompt")
    parser.add_argument("--duration", type=int, default=60,
                        help="Target video duration in seconds (default: 60)")
    parser.add_argument("--tone", default="educational",
                        choices=["educational", "cinematic", "promotional"],
                        help="Video tone (default: educational)")
    parser.add_argument("--language", default="en",
                        help="ISO language code (default: en)")
    parser.add_argument("--no-animate", action="store_true",
                        help="Skip Runway animation — use static images only (faster, cheaper)")
    parser.add_argument("--storyboard-file", default=None,
                        help="Path to pre-defined storyboard JSON (skips script+storyboard generation)")
    args = parser.parse_args()

    if args.no_animate:
        os.environ["VIDEO_BACKEND"] = "none"

    # Import after env override so settings pick up VIDEO_BACKEND=none
    from pipeline.orchestrator import run_pipeline

    print(f"\nAI Video Production Pipeline")
    print(f"Topic    : {args.topic}")
    print(f"Duration : {args.duration}s")
    print(f"Tone     : {args.tone}")
    print(f"Animate  : {'No (static images)' if args.no_animate else 'Yes (Runway)'}")
    if args.storyboard_file:
        print(f"Storyboard: {args.storyboard_file} (pre-defined, skipping generation)")
    print("-" * 50)

    package = run_pipeline(
        topic=args.topic,
        duration=args.duration,
        tone=args.tone,
        language=args.language,
        storyboard_path=args.storyboard_file,
    )

    print(f"\nDone! Review your preview before publishing:")
    print(f"  Preview : {package.preview_video_path}")
    print(f"  Final   : {package.final_video_path}\n")


if __name__ == "__main__":
    main()
