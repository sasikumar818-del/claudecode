"""Local TTS Voiceover — offline speech synthesis via pyttsx3.

On Linux, install the speech engine first:
    sudo apt-get install espeak espeak-data libespeak1

On macOS/Windows, pyttsx3 uses the built-in TTS engine — no extra install.

Set in .env:
    USE_LOCAL_TTS=true
    LOCAL_TTS_RATE=150   # words per minute (optional)

For higher-quality voices, install Coqui TTS:
    pip install TTS
and see the CoquiTTSVoiceover class at the bottom of this file.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from plugins.base import AgentPlugin
from models.production import AudioAsset
from models.storyboard import Storyboard

# Module-level engine singleton — avoids reloading pyttsx3 on every job.
_ENGINE = None


def _get_engine(rate: int = 150):
    global _ENGINE
    if _ENGINE is None:
        import pyttsx3
        _ENGINE = pyttsx3.init()
    _ENGINE.setProperty("rate", rate)
    return _ENGINE


class LocalTTSVoiceover(AgentPlugin):
    """Replaces ElevenLabs with the system's built-in TTS engine (pyttsx3)."""

    @property
    def name(self) -> str:
        return "local_tts_voiceover"

    @property
    def replaces(self) -> str:
        return "Voiceover"

    def setup(self, settings: Any) -> None:
        self._rate = getattr(settings, "local_tts_rate", 150)
        self._output_dir = settings.output_subdirs["audio"]
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Pre-initialise so the first job doesn't pay the init cost.
        try:
            _get_engine(self._rate)
            print(f"  [LocalTTS] pyttsx3 engine ready (rate={self._rate} wpm)")
        except Exception as exc:
            print(f"  [LocalTTS] pyttsx3 init warning: {exc}")

    def run(self, input_data: Storyboard) -> list[AudioAsset]:
        engine = _get_engine(self._rate)
        assets: list[AudioAsset] = []

        for scene in input_data.scenes:
            print(f"  [LocalTTS] Synthesising scene {scene.scene_id}…")
            wav_path = self._output_dir / f"scene_{scene.scene_id:03d}.wav"
            engine.save_to_file(scene.narration_text, str(wav_path))
            engine.runAndWait()

            duration = _wav_duration(wav_path) or scene.duration_seconds
            assets.append(
                AudioAsset(
                    scene_id=scene.scene_id,
                    file_path=wav_path,
                    duration_seconds=duration,
                    voice_id="local_pyttsx3",
                )
            )
            print(f"  [LocalTTS] Saved {wav_path} ({duration:.1f}s)")

        return assets


def _wav_duration(path: Path) -> float:
    """Return the duration of a WAV file in seconds, or 0 on error."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0
