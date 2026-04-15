"""Video Assembler Agent — animates images via Runway and stitches the final MP4."""
from __future__ import annotations

import base64
import time
from pathlib import Path

from config.settings import Settings, get_settings
from models.production import AudioAsset, ImageAsset, ProductionPackage, VideoClip


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
                clip_path = self._image_to_static_clip(image, audio)
                source = "static"

            package.video_clips.append(
                VideoClip(
                    scene_id=scene_id,
                    file_path=clip_path,
                    duration_seconds=audio.duration_seconds,
                    source=source,
                )
            )

    def _animate_with_runway(self, image: ImageAsset, audio: AudioAsset) -> Path:
        import requests
        from moviepy import VideoFileClip

        clip_path = self.clips_dir / f"scene_{image.scene_id:03d}_runway.mp4"
        # Runway max is 10s; we request 10s and loop the clip to match full audio duration
        runway_duration = min(10, max(1, round(audio.duration_seconds)))
        image_b64 = base64.b64encode(image.file_path.read_bytes()).decode()

        headers = {
            "Authorization": f"Bearer {self.runway_api_key}",
            "X-Runway-Version": "2024-11-06",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gen3a_turbo",
            "promptImage": f"data:image/png;base64,{image_b64}",
            "duration": runway_duration,
            "ratio": f"{self.video_width}:{self.video_height}",
        }

        resp = requests.post(
            f"{self.runway_api_base}/image_to_video",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        task_id = resp.json()["id"]

        # Poll until complete
        raw_path = self.clips_dir / f"scene_{image.scene_id:03d}_runway_raw.mp4"
        for _ in range(60):  # max 10 min
            time.sleep(10)
            poll = requests.get(
                f"{self.runway_api_base}/tasks/{task_id}",
                headers=headers,
                timeout=30,
            )
            poll.raise_for_status()
            status = poll.json().get("status")
            if status == "SUCCEEDED":
                video_url = poll.json()["output"][0]
                video_bytes = requests.get(video_url, timeout=120).content
                raw_path.write_bytes(video_bytes)
                print(f"  [Assembler] Runway raw clip downloaded → {raw_path}")
                break
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Runway task {task_id} failed: {status}")
        else:
            raise TimeoutError(f"Runway task {task_id} timed out after 10 minutes")

        # Loop the Runway clip to match full audio duration, then write final clip
        animated = VideoFileClip(str(raw_path))
        if audio.duration_seconds > runway_duration:
            animated = animated.loop(duration=audio.duration_seconds)
        animated.write_videofile(
            str(clip_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "18", "-preset", "slow"],
            logger=None,
        )
        animated.close()
        raw_path.unlink(missing_ok=True)
        print(f"  [Assembler] Runway clip saved → {clip_path}")
        return clip_path

    def _image_to_static_clip(self, image: ImageAsset, audio: AudioAsset) -> Path:
        from moviepy import AudioFileClip, ImageClip

        clip_path = self.clips_dir / f"scene_{image.scene_id:03d}_static.mp4"
        img_clip = (
            ImageClip(str(image.file_path), duration=audio.duration_seconds)
            .resized((self.video_width, self.video_height))
        )
        audio_clip = AudioFileClip(str(audio.file_path))
        final = img_clip.with_audio(audio_clip)
        final.write_videofile(
            str(clip_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "18", "-preset", "slow"],
            logger=None,
        )
        final.close()
        print(f"  [Assembler] Static clip saved → {clip_path}")
        return clip_path

    # ------------------------------------------------------------------
    # Final stitch
    # ------------------------------------------------------------------

    def _stitch_final(self, package: ProductionPackage) -> None:
        from moviepy import CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips

        sorted_clips = sorted(package.video_clips, key=lambda c: c.scene_id)
        scenes_by_id = {s.scene_id: s for s in package.storyboard.scenes}

        video_clips = [VideoFileClip(str(c.file_path)) for c in sorted_clips]
        composite_clips = []
        offset = 0.0

        for clip, vc in zip(video_clips, sorted_clips):
            scene = scenes_by_id.get(vc.scene_id)
            layers = [clip]

            if scene and scene.on_screen_text:
                text = (
                    TextClip(
                        text=scene.on_screen_text,
                        font_size=32,
                        color="white",
                        stroke_color="black",
                        stroke_width=1,
                    )
                    .with_duration(clip.duration)
                    .with_position(("center", 0.85), relative=True)
                    .with_start(offset)
                )
                layers.append(text)

            composite_clips.extend(layers)
            offset += clip.duration

        final = concatenate_videoclips(video_clips, method="compose")

        run_id = package.pipeline_run_id
        out_path = self.final_dir / f"{run_id}_final.mp4"
        final.write_videofile(
            str(out_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "18", "-preset", "slow"],
            logger=None,
        )
        final.close()
        for c in video_clips:
            c.close()

        package.final_video_path = out_path
        print(f"  [Assembler] Final video saved → {out_path}")
