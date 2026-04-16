"""Voiceover Agent — converts scene narration to MP3.

Backends (set TTS_BACKEND in .env):
  espeak      — espeak-ng offline TTS (FREE, default, no internet/key needed)
                Tamil support: lang='ta', voice='ta-IN-PallaviNeural' for edge-tts
                Requires: sudo apt install espeak-ng  ffmpeg
  gtts        — Google TTS (FREE, needs internet, no API key)
  edge-tts    — Microsoft Edge TTS (FREE, needs internet, no API key, high quality)
                Tamil voice: ta-IN-PallaviNeural (female), ta-IN-ValluvarNeural (male)
  elevenlabs  — ElevenLabs paid API (requires ELEVENLABS_API_KEY)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from config.settings import Settings, get_settings
from models.production import AudioAsset
from models.storyboard import Storyboard


class VoiceoverAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tts_backend = settings.tts_backend
        self.output_dir = settings.output_subdirs["audio"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.tts_backend == "elevenlabs":
            from elevenlabs.client import ElevenLabs
            self.elevenlabs_client = ElevenLabs(
                api_key=settings.elevenlabs_api_key.get_secret_value()
            )
            self.elevenlabs_model_id = settings.elevenlabs_model_id

    def run(self, storyboard: Storyboard) -> list[AudioAsset]:
        language = storyboard.language or "en"
        assets: list[AudioAsset] = []

        for scene in storyboard.scenes:
            print(
                f"  [Voiceover] Scene {scene.scene_id} "
                f"(backend={self.tts_backend}, lang={language})…"
            )
            file_path = self.output_dir / f"scene_{scene.scene_id:03d}.mp3"
            voice_label = self._generate_audio(scene.narration_text, language, file_path)
            duration = self._probe_duration(file_path)
            assets.append(
                AudioAsset(
                    scene_id=scene.scene_id,
                    file_path=file_path,
                    duration_seconds=duration,
                    voice_id=voice_label,
                )
            )
            print(f"  [Voiceover] Saved {file_path} ({duration:.1f}s)")
        return assets

    def _generate_audio(self, text: str, language: str, file_path: Path) -> str:
        if self.tts_backend == "elevenlabs":
            return self._generate_elevenlabs(text, language, file_path)
        if self.tts_backend == "gtts":
            return self._generate_gtts(text, language, file_path)
        if self.tts_backend == "edge-tts":
            return self._generate_edge_tts(text, language, file_path)
        # Default: espeak offline
        return self._generate_espeak(text, language, file_path)

    # ──────────────────────────────────────────────────────────────────
    # espeak-ng — OFFLINE, FREE (default)
    # Install: sudo apt install espeak-ng ffmpeg
    # Supports Tamil with voice code 'ta'
    # ──────────────────────────────────────────────────────────────────
    def _generate_espeak(self, text: str, language: str, file_path: Path) -> str:
        # Map ISO language codes to espeak-ng voice codes
        voice_map = {
            "ta": "ta",       # Tamil
            "en": "en-us",    # English (US)
            "hi": "hi",       # Hindi
            "te": "te",       # Telugu
            "kn": "kn",       # Kannada
            "ml": "ml",       # Malayalam
        }
        espeak_voice = voice_map.get(language, language)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        try:
            # Generate WAV via espeak-ng
            subprocess.run(
                ["espeak-ng", "-v", espeak_voice, "-w", wav_path, text],
                check=True,
                capture_output=True,
            )
            # Convert WAV → MP3 via ffmpeg
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-codec:a", "libmp3lame", "-qscale:a", "2",
                    str(file_path),
                ],
                check=True,
                capture_output=True,
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        return f"espeak:{espeak_voice}"

    # ──────────────────────────────────────────────────────────────────
    # gTTS — Google TTS (FREE, needs internet, no key)
    # ──────────────────────────────────────────────────────────────────
    def _generate_gtts(self, text: str, language: str, file_path: Path) -> str:
        from gtts import gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(str(file_path))
        return f"gtts:{language}"

    # ──────────────────────────────────────────────────────────────────
    # Microsoft Edge TTS — FREE, needs internet, no key
    # Tamil voices: ta-IN-PallaviNeural (female), ta-IN-ValluvarNeural (male)
    # ──────────────────────────────────────────────────────────────────
    def _generate_edge_tts(self, text: str, language: str, file_path: Path) -> str:
        import asyncio
        import edge_tts

        # Map language codes to Edge TTS neural voice names
        voice_map = {
            "ta": "ta-IN-PallaviNeural",
            "en": "en-US-JennyNeural",
            "hi": "hi-IN-SwaraNeural",
            "te": "te-IN-ShrutiNeural",
        }
        voice = voice_map.get(language, "ta-IN-PallaviNeural")

        async def _do_tts():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(file_path))

        asyncio.run(_do_tts())
        return voice

    # ──────────────────────────────────────────────────────────────────
    # ElevenLabs (paid)
    # ──────────────────────────────────────────────────────────────────
    def _generate_elevenlabs(self, text: str, language: str, file_path: Path) -> str:
        voice_id = self.settings.voice_id_for_language(language)
        audio_generator = self.elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self.elevenlabs_model_id,
            output_format="mp3_44100_128",
        )
        file_path.write_bytes(b"".join(audio_generator))
        return voice_id

    def _probe_duration(self, path: Path) -> float:
        try:
            from mutagen.mp3 import MP3
            return MP3(str(path)).info.length
        except Exception:
            # Fallback: estimate from file size (128 kbps)
            size_bits = path.stat().st_size * 8
            return size_bits / 128_000
