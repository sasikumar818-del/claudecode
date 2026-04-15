"""Voiceover Agent — converts scene narration to MP3 via ElevenLabs TTS."""
from __future__ import annotations

from pathlib import Path

from elevenlabs.client import ElevenLabs

from config.settings import Settings, get_settings
from models.production import AudioAsset
from models.storyboard import Storyboard


class VoiceoverAgent:
    def __init__(self, settings: Settings) -> None:
        self.client = ElevenLabs(
            api_key=settings.elevenlabs_api_key.get_secret_value()
        )
        self.settings = settings
        self.model_id = settings.elevenlabs_model_id
        self.output_dir = settings.output_subdirs["audio"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, storyboard: Storyboard) -> list[AudioAsset]:
        language = storyboard.language or "en"
        voice_id = self.settings.voice_id_for_language(language)
        assets: list[AudioAsset] = []
        for scene in storyboard.scenes:
            print(f"  [Voiceover] Generating audio for scene {scene.scene_id} (lang={language})…")
            audio_bytes = self._generate_audio(scene.narration_text, voice_id)
            file_path = self.output_dir / f"scene_{scene.scene_id:03d}.mp3"
            file_path.write_bytes(audio_bytes)
            duration = self._probe_duration(file_path)
            assets.append(
                AudioAsset(
                    scene_id=scene.scene_id,
                    file_path=file_path,
                    duration_seconds=duration,
                    voice_id=voice_id,
                )
            )
            print(f"  [Voiceover] Saved {file_path} ({duration:.1f}s)")
        return assets

    def _generate_audio(self, text: str, voice_id: str) -> bytes:
        audio_generator = self.client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        return b"".join(audio_generator)

    def _probe_duration(self, path: Path) -> float:
        try:
            from mutagen.mp3 import MP3
            return MP3(str(path)).info.length
        except Exception:
            # Fallback: estimate from file size (128kbps)
            size_bits = path.stat().st_size * 8
            return size_bits / 128_000
