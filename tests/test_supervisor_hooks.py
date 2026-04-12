"""Tests for plugin hook firing and agent override in SupervisorAgent."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from models.production import AudioAsset, ImageAsset, ProductionPackage, VideoClip
from models.script import ScriptRequest, ScriptSection, VideoScript
from models.storyboard import Scene, ShotType, Storyboard
from plugins.base import AgentPlugin, BasePlugin
from plugins.registry import PluginRegistry


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.anthropic_api_key.get_secret_value.return_value = "test-key"
    settings.claude_model = "claude-opus-4-5"
    final_dir = tmp_path / "final"
    final_dir.mkdir(parents=True)
    settings.output_subdirs = {
        "scripts": tmp_path / "scripts",
        "storyboards": tmp_path / "storyboards",
        "audio": tmp_path / "audio",
        "images": tmp_path / "images",
        "clips": tmp_path / "clips",
        "final": final_dir,
    }
    return settings


@pytest.fixture
def sample_request():
    return ScriptRequest(topic="Test topic", target_duration_seconds=30)


@pytest.fixture
def sample_script(sample_request):
    return VideoScript(
        request=sample_request,
        title="Test Video",
        sections=[
            ScriptSection(
                section_id=1,
                title="Intro",
                narration_text="Hello world.",
                approximate_duration_seconds=30.0,
            )
        ],
        total_estimated_duration=30.0,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_storyboard(sample_script):
    return Storyboard(
        script=sample_script,
        scenes=[
            Scene(
                scene_id=1,
                section_id=1,
                shot_type=ShotType.WIDE,
                visual_description="A bright stage.",
                camera_direction="static",
                narration_text="Hello world.",
                duration_seconds=30.0,
            )
        ],
    )


@pytest.fixture
def sample_package(sample_storyboard, tmp_path):
    (tmp_path / "audio.mp3").write_bytes(b"")
    (tmp_path / "image.png").write_bytes(b"")
    (tmp_path / "clip.mp4").write_bytes(b"")
    (tmp_path / "final.mp4").write_bytes(b"")
    (tmp_path / "preview.mp4").write_bytes(b"")
    return ProductionPackage(
        storyboard=sample_storyboard,
        audio_assets=[
            AudioAsset(
                scene_id=1,
                file_path=tmp_path / "audio.mp3",
                duration_seconds=30.0,
                voice_id="v1",
            )
        ],
        image_assets=[
            ImageAsset(
                scene_id=1,
                file_path=tmp_path / "image.png",
                prompt_used="A bright stage.",
                backend="dalle",
            )
        ],
        video_clips=[
            VideoClip(
                scene_id=1,
                file_path=tmp_path / "clip.mp4",
                duration_seconds=30.0,
                source="static",
            )
        ],
        final_video_path=tmp_path / "final.mp4",
        preview_video_path=tmp_path / "preview.mp4",
        pipeline_run_id="test_run",
        created_at=datetime.utcnow(),
    )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_supervisor(mock_settings, registry=None):
    """Build a SupervisorAgent with all sub-agents mocked."""
    with (
        patch("agents.supervisor.ScriptWriterAgent"),
        patch("agents.supervisor.StoryboardAgent"),
        patch("agents.supervisor.VoiceoverAgent"),
        patch("agents.supervisor.ImageGeneratorAgent"),
        patch("agents.supervisor.VideoAssemblerAgent"),
        patch("agents.supervisor.ReviewAgent"),
        patch("agents.supervisor.anthropic.Anthropic"),
    ):
        from agents.supervisor import SupervisorAgent
        return SupervisorAgent(mock_settings, registry=registry)


def _wire_agents(supervisor, sample_script, sample_storyboard, sample_package):
    """Attach mock return values to a supervisor's built-in agents."""
    supervisor.script_writer.run.return_value = sample_script
    supervisor.storyboard.run.return_value = sample_storyboard
    supervisor.voiceover.run.return_value = sample_package.audio_assets
    supervisor.image_generator.run.return_value = sample_package.image_assets
    supervisor.video_assembler.run.return_value = sample_package
    supervisor.review.run.return_value = sample_package
    supervisor._supervisor_decision = MagicMock(
        return_value=("proceed", "looks good", "")
    )


# ── Backward compatibility ─────────────────────────────────────────────────────

def test_no_registry_constructs_without_error(mock_settings):
    """Existing callers that pass no registry should work identically."""
    supervisor = _make_supervisor(mock_settings, registry=None)
    # Uses an internal empty registry by default
    assert supervisor._registry is not None
    assert supervisor._registry.all_plugins() == []


# ── Hook invocation ────────────────────────────────────────────────────────────

def test_on_step_start_called_for_each_step(
    mock_settings, sample_request, sample_script, sample_storyboard, sample_package
):
    hook = MagicMock(spec=BasePlugin)
    hook.name = "hook_logger"
    hook.setup = MagicMock()
    hook.teardown = MagicMock()
    hook.on_step_start = MagicMock()
    hook.on_step_end = MagicMock()

    registry = PluginRegistry()
    registry._plugins["hook_logger"] = hook

    supervisor = _make_supervisor(mock_settings, registry=registry)
    _wire_agents(supervisor, sample_script, sample_storyboard, sample_package)
    supervisor.run(sample_request)

    # 6 pipeline steps → 6 on_step_start calls
    assert hook.on_step_start.call_count == 6
    step_names_called = [c.args[0] for c in hook.on_step_start.call_args_list]
    assert "Script Writer" in step_names_called
    assert "Storyboard" in step_names_called
    assert "Voiceover" in step_names_called
    assert "Image Generator" in step_names_called
    assert "Video Assembler" in step_names_called
    assert "Review" in step_names_called


def test_on_step_end_called_for_each_step(
    mock_settings, sample_request, sample_script, sample_storyboard, sample_package
):
    hook = MagicMock(spec=BasePlugin)
    hook.name = "hook_logger"
    hook.setup = MagicMock()
    hook.teardown = MagicMock()
    hook.on_step_start = MagicMock()
    hook.on_step_end = MagicMock()

    registry = PluginRegistry()
    registry._plugins["hook_logger"] = hook

    supervisor = _make_supervisor(mock_settings, registry=registry)
    _wire_agents(supervisor, sample_script, sample_storyboard, sample_package)
    supervisor.run(sample_request)

    assert hook.on_step_end.call_count == 6


def test_hook_exception_does_not_abort_pipeline(
    mock_settings, sample_request, sample_script, sample_storyboard, sample_package
):
    """A hook that raises must not crash the pipeline."""
    bad_hook = MagicMock(spec=BasePlugin)
    bad_hook.name = "bad_hook"
    bad_hook.setup = MagicMock()
    bad_hook.teardown = MagicMock()
    bad_hook.on_step_start.side_effect = RuntimeError("hook exploded")
    bad_hook.on_step_end = MagicMock()

    registry = PluginRegistry()
    registry._plugins["bad_hook"] = bad_hook

    supervisor = _make_supervisor(mock_settings, registry=registry)
    _wire_agents(supervisor, sample_script, sample_storyboard, sample_package)

    # Pipeline should complete despite the hook error
    result = supervisor.run(sample_request)
    assert result is not None


# ── Agent override ─────────────────────────────────────────────────────────────

def test_agent_plugin_replaces_script_writer(
    mock_settings, sample_request, sample_script, sample_storyboard, sample_package
):
    """An AgentPlugin with replaces='Script Writer' should be called instead of built-in."""

    class CustomScriptWriter(AgentPlugin):
        @property
        def name(self) -> str:
            return "custom_script_writer"

        @property
        def replaces(self) -> str:
            return "Script Writer"

        def run(self, input_data: Any) -> Any:
            return sample_script

    registry = PluginRegistry()
    registry.register(CustomScriptWriter())

    supervisor = _make_supervisor(mock_settings, registry=registry)
    _wire_agents(supervisor, sample_script, sample_storyboard, sample_package)

    supervisor.run(sample_request)

    # The built-in script_writer.run should NOT have been called
    supervisor.script_writer.run.assert_not_called()


def test_no_agent_override_uses_builtin(
    mock_settings, sample_request, sample_script, sample_storyboard, sample_package
):
    """Without a matching AgentPlugin, the built-in agent is used."""
    registry = PluginRegistry()  # empty

    supervisor = _make_supervisor(mock_settings, registry=registry)
    _wire_agents(supervisor, sample_script, sample_storyboard, sample_package)
    supervisor.run(sample_request)

    supervisor.script_writer.run.assert_called_once()


def test_resolve_agent_returns_override_when_registered(mock_settings):
    class DummyAgent(AgentPlugin):
        @property
        def name(self) -> str:
            return "dummy_agent"

        @property
        def replaces(self) -> str:
            return "Voiceover"

        def run(self, input_data: Any) -> Any:
            return []

    registry = PluginRegistry()
    registry.register(DummyAgent())

    supervisor = _make_supervisor(mock_settings, registry=registry)
    dummy = registry.get_agent_override("Voiceover")
    assert supervisor._resolve_agent("Voiceover", supervisor.voiceover) is dummy


def test_resolve_agent_returns_fallback_when_not_registered(mock_settings):
    registry = PluginRegistry()
    supervisor = _make_supervisor(mock_settings, registry=registry)
    assert supervisor._resolve_agent("Voiceover", supervisor.voiceover) is supervisor.voiceover


# ── Lifecycle: setup and teardown ──────────────────────────────────────────────

def test_setup_called_during_init(mock_settings):
    lifecycle = []

    class LifecyclePlugin(BasePlugin):
        @property
        def name(self) -> str:
            return "lifecycle"

        def setup(self, settings: Any) -> None:
            lifecycle.append("setup")

    registry = PluginRegistry()
    registry.register(LifecyclePlugin())
    _make_supervisor(mock_settings, registry=registry)

    assert lifecycle == ["setup"]


def test_teardown_called_after_run(
    mock_settings, sample_request, sample_script, sample_storyboard, sample_package
):
    lifecycle = []

    class LifecyclePlugin(BasePlugin):
        @property
        def name(self) -> str:
            return "lifecycle"

        def teardown(self) -> None:
            lifecycle.append("teardown")

    registry = PluginRegistry()
    registry.register(LifecyclePlugin())

    supervisor = _make_supervisor(mock_settings, registry=registry)
    _wire_agents(supervisor, sample_script, sample_storyboard, sample_package)
    supervisor.run(sample_request)

    assert lifecycle == ["teardown"]


def test_teardown_called_even_on_pipeline_failure(mock_settings, sample_request):
    """teardown must run even when the pipeline raises."""
    teardown_called = []

    class CleanupPlugin(BasePlugin):
        @property
        def name(self) -> str:
            return "cleanup"

        def teardown(self) -> None:
            teardown_called.append(True)

    registry = PluginRegistry()
    registry.register(CleanupPlugin())

    supervisor = _make_supervisor(mock_settings, registry=registry)
    supervisor.script_writer.run.side_effect = RuntimeError("step exploded")
    supervisor._supervisor_decision = MagicMock(return_value=("abort", "unrecoverable", ""))

    with pytest.raises(RuntimeError):
        supervisor.run(sample_request)

    assert len(teardown_called) == 1
