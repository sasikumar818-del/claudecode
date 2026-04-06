"""Tests for the Script Writer Agent."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from models.script import ScriptRequest, VideoScript


MOCK_LLM_RESPONSE = json.dumps({
    "title": "The History of Space Exploration",
    "sections": [
        {
            "section_id": 1,
            "title": "The Early Days",
            "narration_text": "In the late 1950s, humanity took its first steps into space.",
            "approximate_duration_seconds": 20.0,
        },
        {
            "section_id": 2,
            "title": "Moon Landing",
            "narration_text": "In 1969, Apollo 11 landed on the Moon.",
            "approximate_duration_seconds": 20.0,
        },
        {
            "section_id": 3,
            "title": "The Future",
            "narration_text": "Today, we set our sights on Mars.",
            "approximate_duration_seconds": 20.0,
        },
    ],
    "total_estimated_duration": 60.0,
})


@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.anthropic_api_key.get_secret_value.return_value = "test-key"
    settings.claude_model = "claude-opus-4-5"
    settings.output_subdirs = {"scripts": tmp_path / "scripts"}
    return settings


@patch("agents.script_writer.anthropic.Anthropic")
def test_run_returns_video_script(mock_anthropic_cls, mock_settings):
    from agents.script_writer import ScriptWriterAgent

    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value.content = [
        MagicMock(text=MOCK_LLM_RESPONSE)
    ]

    agent = ScriptWriterAgent(mock_settings)
    request = ScriptRequest(topic="Space exploration", target_duration_seconds=60)
    result = agent.run(request)

    assert isinstance(result, VideoScript)
    assert result.title == "The History of Space Exploration"
    assert len(result.sections) == 3
    assert result.total_estimated_duration == 60.0


@patch("agents.script_writer.anthropic.Anthropic")
def test_script_saved_to_disk(mock_anthropic_cls, mock_settings, tmp_path):
    from agents.script_writer import ScriptWriterAgent

    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value.content = [
        MagicMock(text=MOCK_LLM_RESPONSE)
    ]

    agent = ScriptWriterAgent(mock_settings)
    request = ScriptRequest(topic="Space exploration")
    agent.run(request)

    scripts_dir = mock_settings.output_subdirs["scripts"]
    saved_files = list(scripts_dir.glob("*_script.json"))
    assert len(saved_files) == 1
