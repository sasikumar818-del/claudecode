"""Tests for the Image Generator Agent."""
from __future__ import annotations

import base64
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.production import ImageAsset
from models.script import ScriptRequest, ScriptSection, VideoScript
from models.storyboard import Scene, ShotType, Storyboard


FAKE_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64).decode()


@pytest.fixture
def mock_settings_dalle(tmp_path):
    settings = MagicMock()
    settings.image_backend = "dalle"
    settings.openai_api_key.get_secret_value.return_value = "test-key"
    settings.output_subdirs = {"images": tmp_path / "images"}
    return settings


@pytest.fixture
def sample_storyboard():
    script = VideoScript(
        request=ScriptRequest(topic="Space"),
        title="Space",
        sections=[
            ScriptSection(
                section_id=1, title="Intro",
                narration_text="Space is vast.",
                approximate_duration_seconds=10.0,
            )
        ],
        total_estimated_duration=10.0,
        created_at=datetime.utcnow(),
    )
    return Storyboard(
        script=script,
        scenes=[
            Scene(
                scene_id=1, section_id=1,
                shot_type=ShotType.WIDE,
                visual_description="A vast galaxy with purple nebulae.",
                camera_direction="static",
                narration_text="Space is vast.",
                duration_seconds=10.0,
            )
        ],
    )


@patch("agents.image_generator.OpenAI")
def test_dalle_generates_image_asset(mock_openai_cls, mock_settings_dalle, sample_storyboard):
    from agents.image_generator import ImageGeneratorAgent

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value.data = [
        MagicMock(b64_json=FAKE_PNG, revised_prompt="A vast galaxy")
    ]

    agent = ImageGeneratorAgent(mock_settings_dalle)
    results = agent.run(sample_storyboard)

    assert len(results) == 1
    asset = results[0]
    assert isinstance(asset, ImageAsset)
    assert asset.scene_id == 1
    assert asset.backend == "dalle"
    assert asset.file_path.suffix == ".png"
    assert asset.revised_prompt == "A vast galaxy"
