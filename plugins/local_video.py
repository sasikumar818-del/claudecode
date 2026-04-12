"""Local Video Assembler — Ken Burns pan-zoom effect, no cloud required.

Replaces the Runway-based VideoAssemblerAgent with a fully local implementation
that applies a smooth pan-and-zoom (Ken Burns) effect to each still image,
then stitches all clips into the final MP4.

Set in .env:
    USE_LOCAL_VIDEO=true

Requires:
    pip install moviepy pillow numpy
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from plugins.base import AgentPlugin
from models.production import ProductionPackage, VideoClip

# Output resolution
_W, _H = 1280, 720
# Scale factor gives headroom for pan/zoom without black borders
_SCALE = 1.14


class LocalKenBurnsAssembler(AgentPlugin):
    """Replaces the Runway-based Video Assembler with a Ken Burns effect pipeline."""

    @property
    def name(self) -> str:
        return "local_ken_burns_assembler"

    @property
    def replaces(self) -> str:
        return "Video Assembler"

    def setup(self, settings: Any) -> None:
        self._clips_dir = settings.output_subdirs["clips"]
        self._final_dir = settings.output_subdirs["final"]
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        self._final_dir.mkdir(parents=True, exist_ok=True)
        print("  [LocalVideo] Ken Burns assembler ready")

    # ------------------------------------------------------------------
    # AgentPlugin.run
    # ------------------------------------------------------------------

    def run(self, input_data: ProductionPackage) -> ProductionPackage:
        self._generate_clips(input_data)
        self._stitch_final(input_data)
        return input_data

    # ------------------------------------------------------------------
    # Per-scene clip generation (Ken Burns)
    # ------------------------------------------------------------------

    def _generate_clips(self, package: ProductionPackage) -> None:
        audio_by_id = {a.scene_id: a for a in package.audio_assets}
        image_by_id = {i.scene_id: i for i in package.image_assets}

        for scene_id in sorted(audio_by_id):
            audio = audio_by_id[scene_id]
            image = image_by_id[scene_id]
            clip_path = self._clips_dir / f"scene_{scene_id:03d}_kenburns.mp4"
            print(f"  [LocalVideo] Rendering Ken Burns clip — scene {scene_id}…")
            _render_ken_burns(image.file_path, audio.file_path, audio.duration_seconds, clip_path)
            package.video_clips.append(
                VideoClip(
                    scene_id=scene_id,
                    file_path=clip_path,
                    duration_seconds=audio.duration_seconds,
                    source="ken_burns",
                )
            )
            print(f"  [LocalVideo] Clip saved → {clip_path}")

    # ------------------------------------------------------------------
    # Final stitch with optional on-screen text
    # ------------------------------------------------------------------

    def _stitch_final(self, package: ProductionPackage) -> None:
        from moviepy import CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips

        sorted_clips = sorted(package.video_clips, key=lambda c: c.scene_id)
        scenes_by_id = {s.scene_id: s for s in package.storyboard.scenes}
        raw_clips = [VideoFileClip(str(c.file_path)) for c in sorted_clips]

        composite_clips = []
        for raw, vc in zip(raw_clips, sorted_clips):
            scene = scenes_by_id.get(vc.scene_id)
            if scene and scene.on_screen_text:
                txt = (
                    TextClip(
                        text=scene.on_screen_text,
                        font_size=32,
                        color="white",
                        stroke_color="black",
                        stroke_width=1,
                    )
                    .with_duration(raw.duration)
                    .with_position(("center", 0.85), relative=True)
                )
                composite_clips.append(CompositeVideoClip([raw, txt]))
            else:
                composite_clips.append(raw)

        final = concatenate_videoclips(composite_clips, method="compose")
        out_path = self._final_dir / f"{package.pipeline_run_id}_final.mp4"
        final.write_videofile(
            str(out_path), fps=24, codec="libx264", audio_codec="aac", logger=None
        )
        final.close()
        for c in raw_clips:
            c.close()

        package.final_video_path = out_path
        print(f"  [LocalVideo] Final video → {out_path}")


# ------------------------------------------------------------------
# Ken Burns rendering helper
# ------------------------------------------------------------------

def _render_ken_burns(
    image_path: Path,
    audio_path: Path,
    duration: float,
    output_path: Path,
) -> None:
    """Render a Ken Burns pan-zoom clip for one scene."""
    from moviepy import AudioFileClip, VideoClip

    # Load and upscale image to give zoom headroom
    img = Image.open(image_path).convert("RGB")
    sw = int(_W * _SCALE)
    sh = int(_H * _SCALE)
    arr = np.array(img.resize((sw, sh), Image.LANCZOS))

    # Pick a random pan direction each time for visual variety
    directions = [
        (0, 0, sw - _W, sh - _H),           # top-left → bottom-right
        (sw - _W, 0, 0, sh - _H),            # top-right → bottom-left
        (0, sh - _H, sw - _W, 0),            # bottom-left → top-right
        ((sw - _W) // 2, 0, (sw - _W) // 2, sh - _H),  # centre → down
    ]
    x0, y0, x1, y1 = random.choice(directions)

    def make_frame(t: float) -> np.ndarray:
        # Smoothstep easing: starts/ends slow, fast in middle
        p = t / duration
        ease = p * p * (3.0 - 2.0 * p)
        x = int(x0 + (x1 - x0) * ease)
        y = int(y0 + (y1 - y0) * ease)
        return arr[y: y + _H, x: x + _W]

    clip = VideoClip(make_frame, duration=duration)
    audio = AudioFileClip(str(audio_path))
    clip = clip.with_audio(audio)
    clip.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    clip.close()
    audio.close()
