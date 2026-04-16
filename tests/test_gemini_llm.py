"""Tests for GeminiScriptWriter plugin."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.script import ScriptRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.gemini_api_key.get_secret_value.return_value = "test-api-key"
    s.gemini_model = "gemini-2.0-flash"
    s.output_subdirs = {"scripts": tmp_path / "scripts"}
    return s


def _valid_script_json(duration: int = 60) -> str:
    return json.dumps({
        "title": "Test Video",
        "sections": [
            {
                "section_id": 1,
                "title": "Intro",
                "narration_text": "Welcome to the test video.",
                "approximate_duration_seconds": duration / 3,
            },
            {
                "section_id": 2,
                "title": "Middle",
                "narration_text": "This is the middle section.",
                "approximate_duration_seconds": duration / 3,
            },
            {
                "section_id": 3,
                "title": "End",
                "narration_text": "Thanks for watching.",
                "approximate_duration_seconds": duration / 3,
            },
        ],
        "total_estimated_duration": float(duration),
    })


def _make_request(duration: int = 60) -> ScriptRequest:
    return ScriptRequest(
        topic="Unit testing",
        target_duration_seconds=duration,
        tone="educational",
        language="en",
    )


# ── Fixture: stub out google.generativeai before importing the plugin ─────────

@pytest.fixture(autouse=True)
def stub_genai():
    """Install a minimal google.generativeai stub so no real SDK is needed."""
    genai_stub = types.ModuleType("google.generativeai")
    google_stub = types.ModuleType("google")
    google_stub.generativeai = genai_stub

    # configure() is a no-op
    genai_stub.configure = MagicMock()
    # GenerativeModel returns a mock whose generate_content we can control
    genai_stub.GenerativeModel = MagicMock()

    sys.modules.setdefault("google", google_stub)
    sys.modules["google.generativeai"] = genai_stub
    yield genai_stub
    # Clean up so other tests aren't affected
    sys.modules.pop("google.generativeai", None)
    sys.modules.pop("google", None)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_returns_video_script(stub_genai, tmp_path):
    """run() returns a VideoScript with the correct title and sections."""
    from plugins.gemini_llm import GeminiScriptWriter

    raw_json = _valid_script_json()
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = raw_json
    stub_genai.GenerativeModel.return_value = mock_model

    plugin = GeminiScriptWriter()
    plugin.setup(_make_settings(tmp_path))
    result = plugin.run(_make_request())

    assert result.title == "Test Video"
    assert len(result.sections) == 3
    assert result.sections[0].section_id == 1


def test_script_json_saved_to_disk(stub_genai, tmp_path):
    """run() saves a JSON file under output/scripts/."""
    from plugins.gemini_llm import GeminiScriptWriter

    raw_json = _valid_script_json()
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = raw_json
    stub_genai.GenerativeModel.return_value = mock_model

    plugin = GeminiScriptWriter()
    plugin.setup(_make_settings(tmp_path))
    plugin.run(_make_request())

    saved_files = list((tmp_path / "scripts").glob("*_script.json"))
    assert len(saved_files) == 1


def test_markdown_fences_stripped(stub_genai, tmp_path):
    """run() strips ```json ... ``` fences before parsing."""
    from plugins.gemini_llm import GeminiScriptWriter

    raw_json = _valid_script_json()
    fenced = f"```json\n{raw_json}\n```"
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = fenced
    stub_genai.GenerativeModel.return_value = mock_model

    plugin = GeminiScriptWriter()
    plugin.setup(_make_settings(tmp_path))
    result = plugin.run(_make_request())

    assert result.title == "Test Video"


def test_plain_fences_stripped(stub_genai, tmp_path):
    """run() also strips ``` (no language tag) fences."""
    from plugins.gemini_llm import GeminiScriptWriter

    raw_json = _valid_script_json()
    fenced = f"```\n{raw_json}\n```"
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = fenced
    stub_genai.GenerativeModel.return_value = mock_model

    plugin = GeminiScriptWriter()
    plugin.setup(_make_settings(tmp_path))
    result = plugin.run(_make_request())

    assert result.title == "Test Video"


def test_setup_raises_without_api_key(stub_genai, tmp_path):
    """setup() raises ValueError when gemini_api_key is None."""
    from plugins.gemini_llm import GeminiScriptWriter

    settings = _make_settings(tmp_path)
    settings.gemini_api_key = None

    plugin = GeminiScriptWriter()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        plugin.setup(settings)


def test_name_and_replaces():
    """Plugin metadata is correct."""
    from plugins.gemini_llm import GeminiScriptWriter

    plugin = GeminiScriptWriter()
    assert plugin.name == "gemini_script_writer"
    assert plugin.replaces == "Script Writer"


def test_total_duration_propagated(stub_genai, tmp_path):
    """total_estimated_duration from JSON is stored on VideoScript."""
    from plugins.gemini_llm import GeminiScriptWriter

    raw_json = _valid_script_json(duration=90)
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = raw_json
    stub_genai.GenerativeModel.return_value = mock_model

    plugin = GeminiScriptWriter()
    plugin.setup(_make_settings(tmp_path))
    result = plugin.run(_make_request(duration=90))

    assert result.total_estimated_duration == pytest.approx(90.0)
