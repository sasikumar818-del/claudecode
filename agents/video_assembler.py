"""Video Assembler Agent — applies Ken Burns cinematic effect and stitches 1080p MP4.

For each scene:
  - Slow zoom-in or zoom-out paired with a pan creates the Ken Burns effect.
  - A unique zoom/pan pattern is cycled across all scenes for visual variety.

Final stitch:
  - Clips are cross-faded (1s overlap) for smooth scene transitions.
  - Tamil / multilingual subtitles are overlaid from on_screen_text.
  - Output is 1920×1080, H.264, AAC, 24fps.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np

from config.settings import Settings
from models.production import AudioAsset, ImageAsset, ProductionPackage, VideoClip

# Ken Burns patterns: alternate zoom direction and pan axis per scene
_KB_PATTERNS = [
    {"zoom": "in",  "pan": "right"},
    {"zoom": "out", "pan": "left"},
    {"zoom": "in",  "pan": "left"},
    {"zoom": "out", "pan": "right"},
    {"zoom": "in",  "pan": "center"},
    {"zoom": "out", "pan": "center"},
    {"zoom": "in",  "pan": "right"},
]

FADE_DURATION = 1.0   # seconds for cross-fade between clips
KB_PADDING    = 0.15  # 15% extra canvas size gives zoom room


class VideoAssemblerAgent:
    def __init__(self, settings: Settings) -> None:
        self.video_backend = settings.video_backend
        self.runway_api_key = (
            settings.runway_api_key.get_secret_value()
            if settings.runway_api_key
            else None
        )
        self.runway_api_base = settings.runway_api_base
        self.clips_dir = settings.output_subdirs["clips"]
        self.final_dir = settings.output_subdirs["final"]
        self.video_width = settings.video_width
        self.video_height = settings.video_height
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def run(self, package: ProductionPackage) -> ProductionPackage:
        self._generate_clips(package)
        self._stitch_final(package)
        return package

    # ------------------------------------------------------------------
    # Clip generation
    # ------------------------------------------------------------------

    def _generate_clips(self, package: ProductionPackage) -> None:
        audio_by_id = {a.scene_id: a for a in package.audio_assets}
        image_by_id = {i.scene_id: i for i in package.image_assets}

        for scene_id in sorted(audio_by_id):
            audio = audio_by_id[scene_id]
            image = image_by_id[scene_id]
            print(f"  [Assembler] Building clip for scene {scene_id}…")

            if self.video_backend == "runway":
                clip_path = self._animate_with_runway(image, audio)
                source = "runway"
            else:
                kb = _KB_PATTERNS[(scene_id - 1) % len(_KB_PATTERNS)]
                clip_path = self._ken_burns_clip(image, audio, kb)
                source = "ken_burns"

            package.video_clips.append(
                VideoClip(
                    scene_id=scene_id,
                    file_path=clip_path,
                    duration_seconds=audio.duration_seconds,
                    source=source,
                )
            )

    # ------------------------------------------------------------------
    # Ken Burns effect (offline, zero-cost)
    # ------------------------------------------------------------------

    def _ken_burns_clip(
        self, image: ImageAsset, audio: AudioAsset, kb: dict
    ) -> Path:
        from PIL import Image
        from moviepy import AudioFileClip, VideoClip

        clip_path = self.clips_dir / f"scene_{image.scene_id:03d}_kb.mp4"
        duration = audio.duration_seconds
        tw, th = self.video_width, self.video_height

        # Upscale source image to give room for zoom/pan
        src = Image.open(str(image.file_path)).convert("RGB")
        base_scale = max(tw / src.width, th / src.height)
        big_scale = base_scale * (1.0 + KB_PADDING)
        sw = max(int(src.width * big_scale), tw + 2)
        sh = max(int(src.height * big_scale), th + 2)
        img_big = np.array(src.resize((sw, sh), Image.LANCZOS))

        zoom_dir = kb["zoom"]   # "in" | "out"
        pan_dir  = kb["pan"]    # "left" | "right" | "center"

        def make_frame(t: float) -> np.ndarray:
            p = t / duration  # 0 → 1

            # Zoom: crop window shrinks (zoom-in) or grows (zoom-out)
            if zoom_dir == "in":
                # start big crop, shrink toward tw×th
                cw = int(tw + (sw - tw) * (1.0 - p))
                ch = int(th + (sh - th) * (1.0 - p))
            else:
                # start tw×th, grow toward big crop
                cw = int(tw + (sw - tw) * p)
                ch = int(th + (sh - th) * p)

            cw = max(min(cw, sw), tw)
            ch = max(min(ch, sh), th)

            max_x = sw - cw
            max_y = sh - ch

            # Pan: horizontal drift
            if pan_dir == "right":
                x = int(max_x * p)
            elif pan_dir == "left":
                x = int(max_x * (1.0 - p))
            else:
                x = max_x // 2

            y = max_y // 2  # vertical center

            crop = img_big[y : y + ch, x : x + cw]
            frame_pil = Image.fromarray(crop).resize((tw, th), Image.LANCZOS)
            return np.array(frame_pil)

        video_clip = VideoClip(make_frame, duration=duration)
        audio_clip = AudioFileClip(str(audio.file_path))
        final = video_clip.with_audio(audio_clip)
        final.write_videofile(
            str(clip_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "20"],
            logger=None,
        )
        final.close()
        audio_clip.close()
        print(f"  [Assembler] Ken Burns clip → {clip_path.name}")
        return clip_path

    # ------------------------------------------------------------------
    # Runway animation (paid, optional)
    # ------------------------------------------------------------------

    def _animate_with_runway(self, image: ImageAsset, audio: AudioAsset) -> Path:
        import requests

        clip_path = self.clips_dir / f"scene_{image.scene_id:03d}_runway.mp4"
        duration = min(10, max(1, round(audio.duration_seconds)))
        image_b64 = base64.b64encode(image.file_path.read_bytes()).decode()

        headers = {
            "Authorization": f"Bearer {self.runway_api_key}",
            "X-Runway-Version": "2024-11-06",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gen3a_turbo",
            "promptImage": f"data:image/png;base64,{image_b64}",
            "duration": duration,
            "ratio": "1280:720",
        }

        resp = requests.post(
            f"{self.runway_api_base}/image_to_video",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        task_id = resp.json()["id"]

        for _ in range(60):
            time.sleep(10)
            poll = requests.get(
                f"{self.runway_api_base}/tasks/{task_id}",
                headers=headers,
                timeout=30,
            )
            poll.raise_for_status()
            status = poll.json().get("status")
            if status == "SUCCEEDED":
                video_bytes = requests.get(poll.json()["output"][0], timeout=120).content
                clip_path.write_bytes(video_bytes)
                return clip_path
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Runway task {task_id} failed: {status}")

        raise TimeoutError(f"Runway task {task_id} timed out after 10 minutes")

    # ------------------------------------------------------------------
    # Final stitch with cross-fades and subtitle overlay
    # ------------------------------------------------------------------

    def _stitch_final(self, package: ProductionPackage) -> None:
        from moviepy import VideoFileClip, concatenate_videoclips
        from moviepy.video.fx import FadeIn, FadeOut, CrossFadeIn

        sorted_clips = sorted(package.video_clips, key=lambda c: c.scene_id)
        scenes_by_id = {s.scene_id: s for s in package.storyboard.scenes}

        raw_clips = [VideoFileClip(str(c.file_path)) for c in sorted_clips]

        # Apply cross-fade: each clip (except first) fades in over previous
        faded: list = []
        for i, clip in enumerate(raw_clips):
            scene = scenes_by_id.get(sorted_clips[i].scene_id)
            c = clip

            if i == 0:
                c = c.with_effects([FadeIn(FADE_DURATION)])
            elif i == len(raw_clips) - 1:
                c = c.with_effects([CrossFadeIn(FADE_DURATION), FadeOut(FADE_DURATION)])
            else:
                c = c.with_effects([CrossFadeIn(FADE_DURATION)])

            # Subtitle overlay
            if scene and scene.on_screen_text:
                try:
                    from moviepy import TextClip, CompositeVideoClip
                    txt = (
                        TextClip(
                            text=scene.on_screen_text,
                            font_size=36,
                            color="white",
                            stroke_color="black",
                            stroke_width=2,
                            method="caption",
                            size=(self.video_width - 80, None),
                        )
                        .with_duration(c.duration)
                        .with_position(("center", 0.88), relative=True)
                    )
                    c = CompositeVideoClip([c, txt])
                except Exception as txt_err:
                    print(f"  [Assembler] Subtitle skipped for scene {sorted_clips[i].scene_id}: {txt_err}")

            faded.append(c)

        final = concatenate_videoclips(faded, padding=-FADE_DURATION, method="compose")

        run_id = package.pipeline_run_id
        out_path = self.final_dir / f"{run_id}_final.mp4"
        final.write_videofile(
            str(out_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "18"],
            logger=None,
        )
        final.close()
        for c in raw_clips:
            c.close()

        package.final_video_path = out_path
        print(f"  [Assembler] Final video → {out_path}")
