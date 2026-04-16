"""Tests for ImagineProVideoAssembler plugin."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.production import AudioAsset, ImageAsset, ProductionPackage
from models.script import ScriptRequest, ScriptSection, VideoScript
from models.storyboard import Scene, ShotType, Storyboard


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.imaginepro_api_key.get_secret_value.return_value = "test-api-key"
    s.imaginepro_api_base = "https://api.imaginepro.ai/api/v1"
    s.output_subdirs = {
        "clips": tmp_path / "clips",
        "final": tmp_path / "final",
    }
    return s


def _make_script() -> VideoScript:
    req = ScriptRequest(topic="Test", target_duration_seconds=30)
    return VideoScript(
        request=req,
        title="Test Script",
        sections=[
            ScriptSection(
                section_id=1,
                title="S1",
                narration_text="Narration.",
                approximate_duration_seconds=30.0,
            )
        ],
        total_estimated_duration=30.0,
    )


def _make_package(tmp_path: Path, num_scenes: int = 2) -> ProductionPackage:
    scenes = [
        Scene(
            scene_id=i,
            section_id=i,
            shot_type=ShotType.WIDE,
            visual_description=f"Visual {i}",
            camera_direction="Static",
            narration_text=f"Narration {i}",
            duration_seconds=5.0,
        )
        for i in range(1, num_scenes + 1)
    ]
    storyboard = Storyboard(script=_make_script(), scenes=scenes)

    # Create dummy image files on disk
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_assets = []
    for i in range(1, num_scenes + 1):
        img_path = images_dir / f"scene_{i:03d}.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # fake PNG
        image_assets.append(
            ImageAsset(
                scene_id=i,
                file_path=img_path,
                prompt_used=f"Visual {i}",
                backend="imaginepro",
            )
        )

    # Create dummy audio files on disk
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_assets = []
    for i in range(1, num_scenes + 1):
        audio_path = audio_dir / f"scene_{i:03d}.wav"
        audio_path.write_bytes(b"RIFF" + b"\x00" * 40)  # fake WAV
        audio_assets.append(
            AudioAsset(
                scene_id=i,
                file_path=audio_path,
                duration_seconds=5.0,
                voice_id="local",
            )
        )

    return ProductionPackage(
        storyboard=storyboard,
        image_assets=image_assets,
        audio_assets=audio_assets,
        pipeline_run_id="test-run-001",
    )


def _mock_http_round_trip(mp4_bytes: bytes = b"FakeMP4"):
    """Return side_effect list for requests.post + requests.get calls per scene."""
    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "vid-task"}}
    submit_resp.raise_for_status = MagicMock()

    poll_done = MagicMock()
    poll_done.json.return_value = {"result": {"status": "completed", "url": "https://cdn.example.com/clip.mp4"}}
    poll_done.raise_for_status = MagicMock()

    dl_resp = MagicMock()
    dl_resp.content = mp4_bytes
    dl_resp.raise_for_status = MagicMock()

    return submit_resp, [poll_done, dl_resp]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_appends_video_clips(tmp_path):
    """run() appends one VideoClip per scene to package.video_clips."""
    from plugins.imaginepro_video import ImagineProVideoAssembler

    plugin = ImagineProVideoAssembler()
    plugin.setup(_make_settings(tmp_path))
    package = _make_package(tmp_path, num_scenes=2)

    submit_resp, get_side = _mock_http_round_trip()
    # Two scenes → two sets of poll+download
    get_sides = get_side * 2

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", side_effect=get_sides), \
         patch.object(plugin, "_stitch_final"):  # skip MoviePy in unit test
        plugin.run(package)

    assert len(package.video_clips) == 2
    assert package.video_clips[0].scene_id == 1
    assert package.video_clips[1].scene_id == 2
    assert package.video_clips[0].source == "imaginepro"


def test_clip_files_saved_to_disk(tmp_path):
    """run() writes MP4 bytes to the clips directory."""
    from plugins.imaginepro_video import ImagineProVideoAssembler

    plugin = ImagineProVideoAssembler()
    plugin.setup(_make_settings(tmp_path))
    package = _make_package(tmp_path, num_scenes=1)

    mp4_data = b"FakeMP4Content"
    submit_resp, get_sides = _mock_http_round_trip(mp4_data)

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", side_effect=get_sides), \
         patch.object(plugin, "_stitch_final"):
        plugin.run(package)

    clip_path = tmp_path / "clips" / "scene_001_imaginepro.mp4"
    assert clip_path.exists()
    assert clip_path.read_bytes() == mp4_data


def test_raises_on_failed_task(tmp_path):
    """_generate_video_clip raises RuntimeError when task status is 'failed'."""
    from plugins.imaginepro_video import ImagineProVideoAssembler

    plugin = ImagineProVideoAssembler()
    plugin.setup(_make_settings(tmp_path))
    package = _make_package(tmp_path, num_scenes=1)

    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "vid-fail"}}
    submit_resp.raise_for_status = MagicMock()

    poll_fail = MagicMock()
    poll_fail.json.return_value = {"result": {"status": "failed", "failReason": "unsupported format"}}
    poll_fail.raise_for_status = MagicMock()

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", return_value=poll_fail), \
         patch("time.sleep"):
        with pytest.raises(RuntimeError, match="unsupported format"):
            plugin.run(package)


def test_raises_on_timeout(tmp_path):
    """_generate_video_clip raises TimeoutError when task never completes."""
    from plugins.imaginepro_video import ImagineProVideoAssembler

    plugin = ImagineProVideoAssembler()
    plugin.setup(_make_settings(tmp_path))
    package = _make_package(tmp_path, num_scenes=1)

    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "vid-slow"}}
    submit_resp.raise_for_status = MagicMock()

    poll_pending = MagicMock()
    poll_pending.json.return_value = {"result": {"status": "processing"}}
    poll_pending.raise_for_status = MagicMock()

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", return_value=poll_pending), \
         patch("time.sleep"), \
         patch("plugins.imaginepro_video._MAX_POLLS", 2):
        with pytest.raises(TimeoutError):
            plugin.run(package)


def test_setup_raises_without_api_key(tmp_path):
    """setup() raises ValueError when imaginepro_api_key is None."""
    from plugins.imaginepro_video import ImagineProVideoAssembler

    settings = _make_settings(tmp_path)
    settings.imaginepro_api_key = None

    plugin = ImagineProVideoAssembler()
    with pytest.raises(ValueError, match="IMAGINEPRO_API_KEY"):
        plugin.setup(settings)


def test_name_and_replaces():
    """Plugin metadata is correct."""
    from plugins.imaginepro_video import ImagineProVideoAssembler

    plugin = ImagineProVideoAssembler()
    assert plugin.name == "imaginepro_video_assembler"
    assert plugin.replaces == "Video Assembler"


def test_run_returns_package(tmp_path):
    """run() returns the same ProductionPackage object it received."""
    from plugins.imaginepro_video import ImagineProVideoAssembler

    plugin = ImagineProVideoAssembler()
    plugin.setup(_make_settings(tmp_path))
    package = _make_package(tmp_path, num_scenes=1)

    submit_resp, get_sides = _mock_http_round_trip()

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", side_effect=get_sides), \
         patch.object(plugin, "_stitch_final"):
        result = plugin.run(package)

    assert result is package
