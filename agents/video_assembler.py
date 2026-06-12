"""Video Assembler Agent — cinematic Ken Burns + cross-fade assembly.

All heavy lifting is delegated to ffmpeg:
  - zoompan filter   → Ken Burns slow zoom + pan (fast, hardware-accelerated)
  - xfade filter     → smooth 1-second cross-fade between every scene
  - acrossfade       → matching audio cross-fade

This is dramatically faster than frame-by-frame Python rendering (~10× speedup
for 1080p content), enabling the full 10-minute video to render in minutes.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from config.settings import Settings
from models.production import AudioAsset, ImageAsset, ProductionPackage, VideoClip

# 7 distinct Ken Burns patterns — different zoom direction + pan axis per scene
_KB_PATTERNS = [
    {"zoom": "in",  "pan": "right"},
    {"zoom": "out", "pan": "left"},
    {"zoom": "in",  "pan": "left"},
    {"zoom": "out", "pan": "right"},
    {"zoom": "in",  "pan": "center"},
    {"zoom": "out", "pan": "center"},
    {"zoom": "in",  "pan": "right"},
]

FADE = 1.0   # cross-fade duration in seconds
ZOOM_RANGE = 0.25   # zoom from 1.0→1.25 (in) or 1.25→1.0 (out)


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
        self.video_width  = settings.video_width
        self.video_height = settings.video_height
        self.ffmpeg_path  = settings.ffmpeg_path
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def run(self, package: ProductionPackage) -> ProductionPackage:
        self._generate_clips(package)
        self._stitch_final(package)
        return package

    # ------------------------------------------------------------------
    # Per-scene clip generation
    # ------------------------------------------------------------------

    def _generate_clips(self, package: ProductionPackage) -> None:
        audio_by_id = {a.scene_id: a for a in package.audio_assets}
        image_by_id = {i.scene_id: i for i in package.image_assets}

        for scene_id in sorted(audio_by_id):
            audio = audio_by_id[scene_id]
            image = image_by_id[scene_id]
            print(f"  [Assembler] Building clip for scene {scene_id}…")

            if self.video_backend == "runway":
                import base64, time, requests
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
    # Ken Burns via ffmpeg zoompan (fast)
    # ------------------------------------------------------------------

    def _ken_burns_clip(
        self, image: ImageAsset, audio: AudioAsset, kb: dict
    ) -> Path:
        clip_path = self.clips_dir / f"scene_{image.scene_id:03d}_kb.mp4"
        duration  = audio.duration_seconds
        fps       = 24
        d_frames  = int(duration * fps) + 1

        W, H = self.video_width, self.video_height
        # Scale source to 130% to give zoom headroom (must be even numbers)
        sw = int(W * 1.3) + (int(W * 1.3) % 2)
        sh = int(H * 1.3) + (int(H * 1.3) % 2)

        zoom_dir = kb["zoom"]
        pan_dir  = kb["pan"]

        if zoom_dir == "in":
            z_expr = f"1+{ZOOM_RANGE}*on/duration"
        else:
            z_expr = f"1+{ZOOM_RANGE}-{ZOOM_RANGE}*on/duration"

        if pan_dir == "right":
            x_expr = "iw/2-(iw/zoom/2)+(iw-iw/zoom)*(on/duration)/2"
        elif pan_dir == "left":
            x_expr = "iw/2-(iw/zoom/2)+(iw-iw/zoom)*(1-on/duration)/2"
        else:  # center
            x_expr = "iw/2-(iw/zoom/2)"

        y_expr = "ih/2-(ih/zoom/2)"

        zoompan = (
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={d_frames}:s={W}x{H}:fps={fps}"
        )
        vf = f"scale={sw}:{sh}:flags=lanczos,{zoompan},format=yuv420p"

        cmd = [
            self.ffmpeg_path, "-y",
            "-loop", "1",
            "-framerate", str(fps),
            "-i", str(image.file_path),
            "-i", str(audio.file_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(duration),
            "-shortest",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg zoompan failed for scene {image.scene_id}:\n{result.stderr[-800:]}"
            )
        print(f"  [Assembler] Ken Burns clip → {clip_path.name}")
        return clip_path

    # ------------------------------------------------------------------
    # Final stitch with cross-fades via ffmpeg xfade + acrossfade
    # ------------------------------------------------------------------

    # Tamil subtitle font — installed alongside Noto fonts
    _SUBTITLE_FONT = "/usr/share/fonts/truetype/noto/NotoSansTamilUI-Regular.ttf"
    _FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    def _stitch_final(self, package: ProductionPackage) -> None:
        sorted_clips = sorted(package.video_clips, key=lambda c: c.scene_id)
        scenes_by_id = {s.scene_id: s for s in package.storyboard.scenes}
        clip_paths   = [c.file_path for c in sorted_clips]
        durations    = [c.duration_seconds for c in sorted_clips]

        run_id   = package.pipeline_run_id
        out_path = self.final_dir / f"{run_id}_final.mp4"

        if len(clip_paths) == 1:
            import shutil
            shutil.copy(str(clip_paths[0]), str(out_path))
            package.final_video_path = out_path
            return

        n = len(clip_paths)

        # Cumulative offset for each xfade transition
        # offset_i = sum(d[0..i]) - FADE * (i+1)
        offsets: list[float] = []
        cumulative = 0.0
        for i in range(n - 1):
            cumulative += durations[i]
            offsets.append(max(0.0, cumulative - FADE * (i + 1)))

        # Build filter_complex string
        fc: list[str] = []

        # Video xfade chain
        prev_v = "0:v"
        for i in range(n - 1):
            label = f"v{i+1}"
            fc.append(
                f"[{prev_v}][{i+1}:v]"
                f"xfade=transition=fade:duration={FADE:.3f}:offset={offsets[i]:.3f}"
                f"[{label}]"
            )
            prev_v = label

        # Audio acrossfade chain
        prev_a = "0:a"
        for i in range(n - 1):
            label = f"a{i+1}"
            fc.append(
                f"[{prev_a}][{i+1}:a]acrossfade=d={FADE:.3f}[{label}]"
            )
            prev_a = label

        # Build subtitle drawtext filters chained onto the final video stream
        import os
        font_path = (self._SUBTITLE_FONT
                     if os.path.exists(self._SUBTITLE_FONT)
                     else self._FALLBACK_FONT)

        # Calculate absolute start/end times for each scene in the merged timeline
        scene_times: list[tuple[float, float]] = []
        t = 0.0
        for i, d in enumerate(durations):
            start = max(0.0, t - FADE * i)
            end   = start + d - (FADE if i < n - 1 else 0)
            scene_times.append((start, end))
            t += d

        subtitle_vf: list[str] = []
        for i, vc in enumerate(sorted_clips):
            scene = scenes_by_id.get(vc.scene_id)
            text  = (scene.on_screen_text if scene and scene.on_screen_text else "").strip()
            if not text:
                continue
            # Escape special ffmpeg drawtext chars
            safe = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
            t_start, t_end = scene_times[i]
            subtitle_vf.append(
                f"drawtext=fontfile='{font_path}'"
                f":text='{safe}'"
                f":fontcolor=white"
                f":fontsize=38"
                f":borderw=2"
                f":bordercolor=black"
                f":x=(w-text_w)/2"
                f":y=h*0.88"
                f":enable='between(t,{t_start:.2f},{t_end:.2f})'"
            )

        if subtitle_vf:
            fc.append(f"[{prev_v}]{'[tmp];[tmp]'.join(subtitle_vf)}[{prev_v}_sub]")
            prev_v = f"{prev_v}_sub"

        filter_complex = ";".join(fc)

        inputs: list[str] = []
        for p in clip_paths:
            inputs += ["-i", str(p)]

        cmd = [
            self.ffmpeg_path, "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{prev_v}]",
            "-map", f"[{prev_a}]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        print(f"  [Assembler] Stitching {n} clips + subtitles…")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Retry without subtitles if drawtext fails (e.g. complex Tamil escaping)
            print("  [Assembler] Subtitle burn failed, retrying without subtitles…")
            fc_nosub = ";".join(fc[: len(fc) - (1 if subtitle_vf else 0)])
            map_v    = prev_v.replace("_sub", "")
            cmd2 = [
                self.ffmpeg_path, "-y",
                *inputs,
                "-filter_complex", fc_nosub,
                "-map", f"[{map_v}]",
                "-map", f"[{prev_a}]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_path),
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            if result2.returncode != 0:
                raise RuntimeError(f"ffmpeg stitch failed:\n{result2.stderr[-1000:]}")

        package.final_video_path = out_path
        total = sum(durations) - FADE * (n - 1)
        print(f"  [Assembler] Final video → {out_path}  ({total:.0f}s)")

    # ------------------------------------------------------------------
    # Runway animation (paid, optional)
    # ------------------------------------------------------------------

    def _animate_with_runway(self, image: ImageAsset, audio: AudioAsset) -> Path:
        import base64
        import time
        import requests

        clip_path = self.clips_dir / f"scene_{image.scene_id:03d}_runway.mp4"
        duration  = min(10, max(1, round(audio.duration_seconds)))
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

        raise TimeoutError(f"Runway task {task_id} timed out")
