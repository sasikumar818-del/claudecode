"""Tests for ImagineProImageGenerator plugin."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from models.script import ScriptRequest, ScriptSection, VideoScript
from models.storyboard import Scene, ShotType, Storyboard


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.imaginepro_api_key.get_secret_value.return_value = "test-api-key"
    s.imaginepro_api_base = "https://api.imaginepro.ai/api/v1"
    s.output_subdirs = {"images": tmp_path / "images"}
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


def _make_storyboard(num_scenes: int = 2) -> Storyboard:
    scenes = [
        Scene(
            scene_id=i,
            section_id=i,
            shot_type=ShotType.WIDE,
            visual_description=f"Visual description for scene {i}.",
            camera_direction="Static shot",
            narration_text=f"Narration for scene {i}.",
            duration_seconds=10.0,
        )
        for i in range(1, num_scenes + 1)
    ]
    return Storyboard(script=_make_script(), scenes=scenes)


def _png_bytes() -> bytes:
    """Minimal valid PNG bytes (1×1 white pixel)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_returns_image_assets(tmp_path):
    """run() returns one ImageAsset per storyboard scene."""
    from plugins.imaginepro_image import ImagineProImageGenerator

    plugin = ImagineProImageGenerator()
    plugin.setup(_make_settings(tmp_path))

    storyboard = _make_storyboard(2)
    img_data = _png_bytes()

    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "task-abc"}}
    submit_resp.raise_for_status = MagicMock()

    poll_pending = MagicMock()
    poll_pending.json.return_value = {"result": {"status": "processing"}}
    poll_pending.raise_for_status = MagicMock()

    poll_done = MagicMock()
    poll_done.json.return_value = {"result": {"status": "completed", "uri": "https://cdn.example.com/img.png"}}
    poll_done.raise_for_status = MagicMock()

    dl_resp = MagicMock()
    dl_resp.content = img_data
    dl_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=submit_resp) as mock_post, \
         patch("requests.get", side_effect=[poll_pending, poll_done, dl_resp,
                                            poll_pending, poll_done, dl_resp]):
        assets = plugin.run(storyboard)

    assert len(assets) == 2
    assert assets[0].scene_id == 1
    assert assets[1].scene_id == 2
    assert assets[0].backend == "imaginepro"


def test_image_files_saved_to_disk(tmp_path):
    """run() writes PNG files to the output directory."""
    from plugins.imaginepro_image import ImagineProImageGenerator

    plugin = ImagineProImageGenerator()
    plugin.setup(_make_settings(tmp_path))
    storyboard = _make_storyboard(1)
    img_data = _png_bytes()

    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "task-xyz"}}
    submit_resp.raise_for_status = MagicMock()

    poll_done = MagicMock()
    poll_done.json.return_value = {"result": {"status": "completed", "uri": "https://cdn.example.com/img.png"}}
    poll_done.raise_for_status = MagicMock()

    dl_resp = MagicMock()
    dl_resp.content = img_data
    dl_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", side_effect=[poll_done, dl_resp]):
        assets = plugin.run(storyboard)

    saved = tmp_path / "images" / "scene_001.png"
    assert saved.exists()
    assert saved.read_bytes() == img_data
    assert assets[0].file_path == saved


def test_raises_on_failed_task(tmp_path):
    """run() raises RuntimeError when the API task status is 'failed'."""
    from plugins.imaginepro_image import ImagineProImageGenerator

    plugin = ImagineProImageGenerator()
    plugin.setup(_make_settings(tmp_path))
    storyboard = _make_storyboard(1)

    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "task-fail"}}
    submit_resp.raise_for_status = MagicMock()

    poll_fail = MagicMock()
    poll_fail.json.return_value = {"result": {"status": "failed", "failReason": "bad prompt"}}
    poll_fail.raise_for_status = MagicMock()

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", return_value=poll_fail), \
         patch("time.sleep"):
        with pytest.raises(RuntimeError, match="bad prompt"):
            plugin.run(storyboard)


def test_raises_on_timeout(tmp_path):
    """run() raises TimeoutError when the task never completes."""
    from plugins.imaginepro_image import ImagineProImageGenerator

    plugin = ImagineProImageGenerator()
    plugin.setup(_make_settings(tmp_path))
    storyboard = _make_storyboard(1)

    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "task-slow"}}
    submit_resp.raise_for_status = MagicMock()

    poll_pending = MagicMock()
    poll_pending.json.return_value = {"result": {"status": "processing"}}
    poll_pending.raise_for_status = MagicMock()

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", return_value=poll_pending), \
         patch("time.sleep"), \
         patch("plugins.imaginepro_image._MAX_POLLS", 2):
        with pytest.raises(TimeoutError):
            plugin.run(storyboard)


def test_setup_raises_without_api_key(tmp_path):
    """setup() raises ValueError when imaginepro_api_key is None."""
    from plugins.imaginepro_image import ImagineProImageGenerator

    settings = _make_settings(tmp_path)
    settings.imaginepro_api_key = None

    plugin = ImagineProImageGenerator()
    with pytest.raises(ValueError, match="IMAGINEPRO_API_KEY"):
        plugin.setup(settings)


def test_name_and_replaces():
    """Plugin metadata is correct."""
    from plugins.imaginepro_image import ImagineProImageGenerator

    plugin = ImagineProImageGenerator()
    assert plugin.name == "imaginepro_image_generator"
    assert plugin.replaces == "Image Generator"


def test_prompt_stored_on_asset(tmp_path):
    """The visual_description prompt is recorded in the ImageAsset."""
    from plugins.imaginepro_image import ImagineProImageGenerator

    plugin = ImagineProImageGenerator()
    plugin.setup(_make_settings(tmp_path))
    storyboard = _make_storyboard(1)
    img_data = _png_bytes()

    submit_resp = MagicMock()
    submit_resp.json.return_value = {"result": {"taskId": "task-p"}}
    submit_resp.raise_for_status = MagicMock()

    poll_done = MagicMock()
    poll_done.json.return_value = {"result": {"status": "completed", "uri": "https://cdn.example.com/img.png"}}
    poll_done.raise_for_status = MagicMock()

    dl_resp = MagicMock()
    dl_resp.content = img_data
    dl_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=submit_resp), \
         patch("requests.get", side_effect=[poll_done, dl_resp]):
        assets = plugin.run(storyboard)

    assert assets[0].prompt_used == "Visual description for scene 1."
