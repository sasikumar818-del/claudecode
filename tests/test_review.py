"""Tests for the Review Agent."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.production import AudioAsset, ImageAsset, ProductionPackage, VideoClip
from models.script import ScriptRequest, ScriptSection, VideoScript
from models.storyboard import Scene, ShotType, Storyboard


@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.watermark_text = "PREVIEW — NOT FOR DISTRIBUTION"
    settings.output_subdirs = {"final": tmp_path / "final"}
    return settings


@pytest.fixture
def sample_package(tmp_path):
    script = VideoScript(
        request=ScriptRequest(topic="Space"),
        title="Space Exploration",
        sections=[
            ScriptSection(
                section_id=1, title="Intro",
                narration_text="Welcome.",
                approximate_duration_seconds=5.0,
            )
        ],
        total_estimated_duration=5.0,
        created_at=datetime.utcnow(),
    )
    storyboard = Storyboard(
        script=script,
        scenes=[
            Scene(
                scene_id=1, section_id=1,
                shot_type=ShotType.WIDE,
                visual_description="Stars.",
                camera_direction="static",
                narration_text="Welcome.",
                duration_seconds=5.0,
            )
        ],
    )
    final_path = tmp_path / "final_video.mp4"
    final_path.write_bytes(b"fake video content")

    return ProductionPackage(
        storyboard=storyboard,
        audio_assets=[
            AudioAsset(
                scene_id=1,
                file_path=tmp_path / "scene_001.mp3",
                duration_seconds=5.0,
                voice_id="test-voice",
            )
        ],
        image_assets=[
            ImageAsset(
                scene_id=1,
                file_path=tmp_path / "scene_001.png",
                prompt_used="Stars.",
                backend="dalle",
            )
        ],
        video_clips=[
            VideoClip(
                scene_id=1,
                file_path=tmp_path / "scene_001.mp4",
                duration_seconds=5.0,
                source="static",
            )
        ],
        final_video_path=final_path,
        created_at=datetime.utcnow(),
        pipeline_run_id="test_run_id",
    )


@patch("agents.review.VideoFileClip")
@patch("agents.review.CompositeVideoClip")
@patch("agents.review.TextClip")
def test_review_writes_report_and_sets_preview(
    mock_text_cls, mock_composite_cls, mock_video_cls,
    mock_settings, sample_package, tmp_path
):
    from agents.review import ReviewAgent

    # Mock MoviePy chain
    mock_final_clip = MagicMock()
    mock_final_clip.duration = 5.0
    mock_video_cls.return_value = mock_final_clip

    mock_text = MagicMock()
    mock_text.with_opacity.return_value.with_duration.return_value.with_position.return_value = mock_text
    mock_text_cls.return_value = mock_text

    mock_preview_clip = MagicMock()
    mock_composite_cls.return_value = mock_preview_clip

    final_dir = mock_settings.output_subdirs["final"]
    final_dir.mkdir(parents=True, exist_ok=True)

    agent = ReviewAgent(mock_settings)
    result = agent.run(sample_package)

    # Preview path should be set
    assert result.preview_video_path is not None

    # Review report should be written
    report_files = list(final_dir.glob("*_review_report.json"))
    assert len(report_files) == 1
    report = json.loads(report_files[0].read_text())
    assert report["title"] == "Space Exploration"
    assert report["total_scenes"] == 1
    assert report["pipeline_run_id"] == "test_run_id"
