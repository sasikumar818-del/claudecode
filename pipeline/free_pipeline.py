"""Zero-cost production pipeline — no API keys required.

Uses:
  - espeak-ng for Tamil/multilingual TTS (offline)
  - HuggingFace Inference API for images (free tier), PIL as fallback
  - Ken Burns + cross-fades via MoviePy (offline)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import get_settings


def _write_review_report(package, output_dir: Path) -> None:
    """Write a JSON review report without requiring the review agent (no watermark)."""
    report = {
        "pipeline_run_id": package.pipeline_run_id,
        "created_at": package.created_at.isoformat(),
        "final_video_path": str(package.final_video_path),
        "scenes": len(package.storyboard.scenes),
        "audio_files": len(package.audio_assets),
        "image_files": len(package.image_assets),
        "video_clips": len(package.video_clips),
        "total_duration_seconds": sum(c.duration_seconds for c in package.video_clips),
    }
    report_path = output_dir / f"{package.pipeline_run_id}_review.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  [Review] Report saved → {report_path}")
    package.preview_video_path = package.final_video_path


def run_free_pipeline(
    storyboard_path: str,
    language: str = "ta",
    image_backend: str = "huggingface",
    tts_backend: str = "espeak",
    output_dir: str = "output",
    video_width: int = 1920,
    video_height: int = 1080,
    hf_api_key: str = "",
):
    """Run the full video production pipeline without any paid API keys.

    Args:
        storyboard_path: Path to the JSON storyboard file.
        language:        ISO language code for TTS (e.g. "ta", "en", "hi").
        image_backend:   "huggingface" (free online) or "pil" (offline).
        tts_backend:     "espeak" (offline) or "gtts" / "edge-tts" (online, free).
        output_dir:      Root output directory.
        video_width:     Output frame width (default 1920).
        video_height:    Output frame height (default 1080).
        hf_api_key:      HuggingFace API key (optional; anonymous if empty).
    """
    # Set env vars before importing settings so they take effect
    os.environ.setdefault("IMAGE_BACKEND", image_backend)
    os.environ.setdefault("TTS_BACKEND", tts_backend)
    os.environ.setdefault("VIDEO_BACKEND", "none")
    os.environ.setdefault("OUTPUT_DIR", output_dir)
    os.environ["VIDEO_WIDTH"] = str(video_width)
    os.environ["VIDEO_HEIGHT"] = str(video_height)
    if hf_api_key:
        os.environ.setdefault("HUGGINGFACE_API_KEY", hf_api_key)

    # Clear cached settings so the new env vars are picked up
    get_settings.cache_clear()
    settings = get_settings()

    # Ensure output dirs exist
    for path in settings.output_subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 62)
    print("  FREE PIPELINE — Zero-cost AI Video Production")
    print("═" * 62)
    print(f"  Storyboard : {storyboard_path}")
    print(f"  Language   : {language}")
    print(f"  TTS        : {tts_backend}")
    print(f"  Images     : {image_backend}")
    print(f"  Resolution : {video_width}×{video_height}")
    print("─" * 62)

    # ── Step 1: Load storyboard ────────────────────────────────────────
    from models.storyboard import Storyboard
    print("\n  ▶  Loading storyboard…")
    storyboard = Storyboard.from_json_file(Path(storyboard_path))
    storyboard.language = language
    print(f"  ✓  Loaded {len(storyboard.scenes)} scenes: {storyboard.title}")

    # ── Step 2: Voiceover ──────────────────────────────────────────────
    from agents.voiceover import VoiceoverAgent
    print("\n  ▶  Voiceover generation…")
    audio_assets = VoiceoverAgent(settings).run(storyboard)
    print(f"  ✓  {len(audio_assets)} audio files generated.")

    # ── Step 3: Image Generation ───────────────────────────────────────
    from agents.image_generator import ImageGeneratorAgent
    print("\n  ▶  Image generation…")
    image_assets = ImageGeneratorAgent(settings).run(storyboard)
    print(f"  ✓  {len(image_assets)} images generated.")

    # ── Step 4: Video Assembly ─────────────────────────────────────────
    from agents.video_assembler import VideoAssemblerAgent
    from models.production import ProductionPackage
    print("\n  ▶  Video assembly (Ken Burns + cross-fades)…")
    package = ProductionPackage(
        storyboard=storyboard,
        audio_assets=audio_assets,
        image_assets=image_assets,
        created_at=datetime.utcnow(),
    )
    package = VideoAssemblerAgent(settings).run(package)
    print(f"  ✓  Final video: {package.final_video_path}")

    # ── Step 5: Review report ──────────────────────────────────────────
    _write_review_report(package, settings.output_subdirs["final"])

    print("\n" + "═" * 62)
    print("  PIPELINE COMPLETE")
    print("═" * 62)
    total = sum(c.duration_seconds for c in package.video_clips)
    print(f"  Scenes   : {len(package.storyboard.scenes)}")
    print(f"  Duration : {total:.1f}s  ({total/60:.1f} min)")
    print(f"  Output   : {package.final_video_path}")
    print("═" * 62 + "\n")

    return package
