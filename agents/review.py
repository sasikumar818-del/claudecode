"""Review Agent — adds a watermark preview and writes a review report."""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import Settings, get_settings
from models.production import ProductionPackage


class ReviewAgent:
    def __init__(self, settings: Settings) -> None:
        self.watermark_text = settings.watermark_text
        self.final_dir = settings.output_subdirs["final"]
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def run(self, package: ProductionPackage) -> ProductionPackage:
        print("  [Review] Creating watermarked preview…")
        preview_path = self._create_watermarked_preview(package)
        package.preview_video_path = preview_path
        self._write_review_report(package)
        self._print_summary(package)
        return package

    def _create_watermarked_preview(self, package: ProductionPackage) -> Path:
        from moviepy import CompositeVideoClip, TextClip, VideoFileClip

        final_clip = VideoFileClip(str(package.final_video_path))
        watermark = (
            TextClip(
                text=self.watermark_text,
                font_size=28,
                color="white",
                stroke_color="black",
                stroke_width=2,
            )
            .with_opacity(0.65)
            .with_duration(final_clip.duration)
            .with_position(("center", 0.92), relative=True)
        )
        preview_clip = CompositeVideoClip([final_clip, watermark])

        preview_path = self.final_dir / f"{package.pipeline_run_id}_preview.mp4"
        preview_clip.write_videofile(
            str(preview_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        preview_clip.close()
        final_clip.close()
        print(f"  [Review] Preview saved → {preview_path}")
        return preview_path

    def _write_review_report(self, package: ProductionPackage) -> None:
        total_duration = sum(c.duration_seconds for c in package.video_clips)
        report = {
            "pipeline_run_id": package.pipeline_run_id,
            "created_at": package.created_at.isoformat(),
            "title": package.storyboard.script.title,
            "total_scenes": len(package.storyboard.scenes),
            "total_duration_seconds": round(total_duration, 2),
            "audio_files": [str(a.file_path) for a in package.audio_assets],
            "image_files": [str(i.file_path) for i in package.image_assets],
            "clip_files": [str(v.file_path) for v in package.video_clips],
            "final_video": str(package.final_video_path),
            "preview_video": str(package.preview_video_path),
        }
        report_path = self.final_dir / f"{package.pipeline_run_id}_review_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"  [Review] Report saved → {report_path}")

    def _print_summary(self, package: ProductionPackage) -> None:
        total = sum(c.duration_seconds for c in package.video_clips)
        print("\n" + "=" * 60)
        print("  REVIEW SUMMARY")
        print("=" * 60)
        print(f"  Title     : {package.storyboard.script.title}")
        print(f"  Run ID    : {package.pipeline_run_id}")
        print(f"  Scenes    : {len(package.storyboard.scenes)}")
        print(f"  Duration  : {total:.1f}s")
        print(f"  Final     : {package.final_video_path}")
        print(f"  Preview   : {package.preview_video_path}")
        print("=" * 60)
        print("  Inspect the preview video before publishing.\n")
