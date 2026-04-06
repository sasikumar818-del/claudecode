"""Tests for the Storyboard Agent."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.script import ScriptRequest, ScriptSection, VideoScript
from models.storyboard import ShotType, Storyboard


MOCK_STORYBOARD_RESPONSE = json.dumps({
    "scenes": [
        {
            "scene_id": 1,
            "section_id": 1,
            "shot_type": "wide",
            "visual_description": "A vast dark sky with stars, a rocket launching upward.",
            "camera_direction": "slow tilt up",
            "on_screen_text": "The Early Days",
            "narration_text": "In the late 1950s, humanity took its first steps into space.",
            "duration_seconds": 20.0,
        },
        {
            "scene_id": 2,
            "section_id": 2,
            "shot_type": "close_up",
            "visual_description": "An astronaut planting an American flag on the Moon surface.",
            "camera_direction": "static shot",
            "on_screen_text": "",
            "narration_text": "In 1969, Apollo 11 landed on the Moon.",
            "duration_seconds": 20.0,
        },
    ]
})


@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.anthropic_api_key.get_secret_value.return_value = "test-key"
    settings.claude_model = "claude-opus-4-5"
    settings.output_subdirs = {"storyboards": tmp_path / "storyboards"}
    return settings


@pytest.fixture
def sample_script():
    return VideoScript(
        request=ScriptRequest(topic="Space exploration"),
        title="Space Exploration",
        sections=[
            ScriptSection(
                section_id=1,
                title="The Early Days",
                narration_text="In the late 1950s, humanity took its first steps.",
                approximate_duration_seconds=20.0,
            ),
            ScriptSection(
                section_id=2,
                title="Moon Landing",
                narration_text="In 1969, Apollo 11 landed on the Moon.",
                approximate_duration_seconds=20.0,
            ),
        ],
        total_estimated_duration=40.0,
        created_at=datetime.utcnow(),
    )


@patch("agents.storyboard.anthropic.Anthropic")
def test_run_returns_storyboard(mock_anthropic_cls, mock_settings, sample_script):
    from agents.storyboard import StoryboardAgent

    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value.content = [
        MagicMock(text=MOCK_STORYBOARD_RESPONSE)
    ]

    agent = StoryboardAgent(mock_settings)
    result = agent.run(sample_script)

    assert isinstance(result, Storyboard)
    assert len(result.scenes) == 2
    assert result.scenes[0].shot_type == ShotType.WIDE
    assert result.scenes[1].shot_type == ShotType.CLOSE_UP
    assert result.scenes[0].on_screen_text == "The Early Days"
    assert result.scenes[1].on_screen_text == ""
