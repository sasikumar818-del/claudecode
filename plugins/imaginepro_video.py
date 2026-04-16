"""ImaginePro Video Assembler — image-to-video clips via ImaginePro (Kling API).

For each scene, submits the generated still image to ImaginePro's image-to-video
endpoint, polls until the clip is ready, then downloads the MP4.  The clips are
stitched together (with audio and optional on-screen text) using MoviePy — the
same stitch logic used by the local Ken Burns assembler.

Set in .env:
    USE_IMAGINEPRO_VIDEO=true
    IMAGINEPRO_API_KEY=your_api_key_here
    IMAGINEPRO_API_BASE=https://api.imaginepro.ai/api/v1
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from plugins.base import AgentPlugin
from models.production import ProductionPackage, VideoClip

# Poll configuration
_POLL_INTERVAL = 5        # seconds between status checks
_MAX_POLLS = 60           # 60 × 5s = 300s max wait per clip
# ImaginePro / Kling maximum clip duration in seconds
_MAX_CLIP_DURATION = 10


class ImagineProVideoAssembler(AgentPlugin):
    """Replaces the Runway / Ken Burns video assembler with ImaginePro Kling."""

    @property
    def name(self) -> str:
        return "imaginepro_video_assembler"

    @property
    def replaces(self) -> str:
        return "Video Assembler"

    def setup(self, settings: Any) -> None:
        api_key = settings.imaginepro_api_key
        if api_key is None:
            raise ValueError("IMAGINEPRO_API_KEY must be set when USE_IMAGINEPRO_VIDEO=true")
        self._api_key = api_key.get_secret_value()
        self._base_url = settings.imaginepro_api_base.rstrip("/")
        self._clips_dir = settings.output_subdirs["clips"]
        self._final_dir = settings.output_subdirs["final"]
        for d in (self._clips_dir, self._final_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"  [ImaginePro] Video assembler ready (base: {self._base_url})")

    # ------------------------------------------------------------------
    # AgentPlugin.run
    # ------------------------------------------------------------------

    def run(self, input_data: ProductionPackage) -> ProductionPackage:
        self._generate_clips(input_data)
        self._stitch_final(input_data)
        return input_data

    # ------------------------------------------------------------------
    # Per-scene clip generation
    # ------------------------------------------------------------------

    def _generate_clips(self, package: ProductionPackage) -> None:
        audio_by_id = {a.scene_id: a for a in package.audio_assets}
        image_by_id = {i.scene_id: i for i in package.image_assets}

        for scene_id in sorted(audio_by_id):
            audio = audio_by_id[scene_id]
            image = image_by_id[scene_id]
            clip_path = self._clips_dir / f"scene_{scene_id:03d}_imaginepro.mp4"
            print(f"  [ImaginePro] Generating video clip — scene {scene_id}…")
            video_bytes = self._generate_video_clip(image.file_path, audio.duration_seconds)
            clip_path.write_bytes(video_bytes)
            package.video_clips.append(
                VideoClip(
                    scene_id=scene_id,
                    file_path=clip_path,
                    duration_seconds=audio.duration_seconds,
                    source="imaginepro",
                )
            )
            print(f"  [ImaginePro] Clip saved → {clip_path}")

    # ------------------------------------------------------------------
    # Final stitch (same logic as local_video.py)
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
        print(f"  [ImaginePro] Final video → {out_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _generate_video_clip(self, image_path: Path, duration: float) -> bytes:
        import requests

        # Encode image as base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        clip_duration = min(int(duration), _MAX_CLIP_DURATION)

        # Submit image-to-video task
        r = requests.post(
            f"{self._base_url}/kling/image-to-video",
            json={"image": img_b64, "duration": clip_duration},
            headers=self._headers(),
            timeout=30,
        )
        r.raise_for_status()
        task_id = r.json()["result"]["taskId"]
        print(f"    [ImaginePro] Video task submitted: {task_id}")

        # Poll until completed or failed
        for attempt in range(_MAX_POLLS):
            time.sleep(_POLL_INTERVAL)
            r = requests.get(
                f"{self._base_url}/kling/task/{task_id}/fetch",
                headers=self._headers(),
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()["result"]
            status = data.get("status", "")

            if status == "completed":
                video_url = data["url"]
                print(f"    [ImaginePro] Task {task_id} complete, downloading…")
                dl = requests.get(video_url, timeout=120)
                dl.raise_for_status()
                return dl.content

            if status == "failed":
                reason = data.get("failReason", "unknown")
                raise RuntimeError(
                    f"ImaginePro video task {task_id} failed: {reason}"
                )

            print(
                f"    [ImaginePro] Waiting… ({attempt + 1}/{_MAX_POLLS}) "
                f"status={status}"
            )

        raise TimeoutError(
            f"ImaginePro video task {task_id} did not complete within "
            f"{_MAX_POLLS * _POLL_INTERVAL}s"
        )
