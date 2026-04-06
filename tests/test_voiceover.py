"""Tests for the Voiceover Agent."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.production import AudioAsset
from models.script import ScriptRequest, ScriptSection, VideoScript
from models.storyboard import Scene, ShotType, Storyboard


@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.elevenlabs_api_key.get_secret_value.return_value = "test-key"
    settings.elevenlabs_voice_id = "test-voice-id"
    settings.elevenlabs_model_id = "eleven_multilingual_v2"
    settings.output_subdirs = {"audio": tmp_path / "audio"}
    return settings


@pytest.fixture
def sample_storyboard():
    script = VideoScript(
        request=ScriptRequest(topic="Space"),
        title="Space",
        sections=[
            ScriptSection(
                section_id=1,
                title="Intro",
                narration_text="Welcome to space.",
                approximate_duration_seconds=5.0,
            )
        ],
        total_estimated_duration=5.0,
        created_at=datetime.utcnow(),
    )
    return Storyboard(
        script=script,
        scenes=[
            Scene(
                scene_id=1,
                section_id=1,
                shot_type=ShotType.WIDE,
                visual_description="A galaxy.",
                camera_direction="static",
                narration_text="Welcome to space.",
                duration_seconds=5.0,
            )
        ],
    )


@patch("agents.voiceover.ElevenLabs")
def test_run_produces_audio_assets(mock_elevenlabs_cls, mock_settings, sample_storyboard, tmp_path):
    from agents.voiceover import VoiceoverAgent

    mock_client = MagicMock()
    mock_elevenlabs_cls.return_value = mock_client
    # Return fake MP3 bytes (minimal valid MP3 header placeholder)
    mock_client.text_to_speech.convert.return_value = iter([b"\xff\xfb" + b"\x00" * 1024])

    agent = VoiceoverAgent(mock_settings)

    with patch.object(agent, "_probe_duration", return_value=5.0):
        results = agent.run(sample_storyboard)

    assert len(results) == 1
    asset = results[0]
    assert isinstance(asset, AudioAsset)
    assert asset.scene_id == 1
    assert asset.duration_seconds == 5.0
    assert asset.voice_id == "test-voice-id"
    assert asset.file_path.suffix == ".mp3"
