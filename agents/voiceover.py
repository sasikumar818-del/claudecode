"""Voiceover Agent — converts scene narration to audio using configurable backends.

Backends (in order of preference for zero-cost use):
  espeak  — offline, no API key, uses espeak-ng + ffmpeg
  gtts    — Google TTS (free, needs internet, may hit rate limits)
  edge-tts — Microsoft TTS (free, needs internet)
  elevenlabs — paid, best quality
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from config.settings import Settings
from models.production import AudioAsset
from models.storyboard import Storyboard

# espeak-ng language codes
_ESPEAK_VOICE: dict[str, str] = {
    "ta": "ta",
    "en": "en-us",
    "hi": "hi",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "ja": "ja",
    "zh": "zh",
    "ko": "ko",
}


class VoiceoverAgent:
    def __init__(self, settings: Settings) -> None:
        self.tts_backend = settings.tts_backend
        self.output_dir = settings.output_subdirs["audio"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_path = settings.ffmpeg_path

        if self.tts_backend == "elevenlabs":
            if not settings.elevenlabs_api_key:
                raise ValueError("elevenlabs_api_key is required for tts_backend='elevenlabs'")
            from elevenlabs.client import ElevenLabs
            self._el_client = ElevenLabs(
                api_key=settings.elevenlabs_api_key.get_secret_value()
            )
            self._el_voice_id = settings.elevenlabs_voice_id
            self._el_model_id = settings.elevenlabs_model_id

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
            print(f"  [Voiceover] Scene {scene.scene_id} → {self.tts_backend}…")
            file_path = self.output_dir / f"scene_{scene.scene_id:03d}.mp3"
            self._synthesize(scene.narration_text, language, file_path)
            duration = self._probe_duration(file_path)
            assets.append(
                AudioAsset(
                    scene_id=scene.scene_id,
                    file_path=file_path,
                    duration_seconds=duration,
                    voice_id=self.tts_backend,
                )
            )
            print(f"  [Voiceover] Saved {file_path.name} ({duration:.1f}s)")
        return assets

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _synthesize(self, text: str, language: str, out_path: Path) -> None:
        if self.tts_backend == "espeak":
            self._generate_espeak(text, language, out_path)
        elif self.tts_backend == "gtts":
            self._generate_gtts(text, language, out_path)
        elif self.tts_backend == "edge-tts":
            self._generate_edge_tts(text, language, out_path)
        elif self.tts_backend == "elevenlabs":
            self._generate_elevenlabs(text, out_path)
        else:
            raise ValueError(f"Unknown TTS backend: {self.tts_backend}")

    # ------------------------------------------------------------------
    # espeak-ng (offline, zero-cost)
    # ------------------------------------------------------------------

    def _generate_espeak(self, text: str, language: str, out_path: Path) -> None:
        voice = _ESPEAK_VOICE.get(language, "en-us")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "130", "-w", wav_path, text],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                self.ffmpeg_path, "-y", "-i", wav_path,
                "-codec:a", "libmp3lame", "-q:a", "2",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        Path(wav_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # gTTS (online, free, may rate-limit)
    # ------------------------------------------------------------------

    def _generate_gtts(self, text: str, language: str, out_path: Path) -> None:
        from gtts import gTTS
        lang_code = language if len(language) == 2 else "en"
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(str(out_path))

    # ------------------------------------------------------------------
    # edge-tts (online, free Microsoft)
    # ------------------------------------------------------------------

    def _generate_edge_tts(self, text: str, language: str, out_path: Path) -> None:
        import asyncio
        import edge_tts

        voice_map = {
            "ta": "ta-IN-ValluvarNeural",
            "en": "en-US-AriaNeural",
            "hi": "hi-IN-SwaraNeural",
        }
        voice = voice_map.get(language, "en-US-AriaNeural")

        async def _run() -> None:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(out_path))

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # ElevenLabs (paid, best quality)
    # ------------------------------------------------------------------

    def _generate_elevenlabs(self, text: str, out_path: Path) -> None:
        chunks = self._el_client.text_to_speech.convert(
            voice_id=self._el_voice_id,
            text=text,
            model_id=self._el_model_id,
            output_format="mp3_44100_128",
        )
        out_path.write_bytes(b"".join(chunks))

    # ------------------------------------------------------------------
    # Duration probe
    # ------------------------------------------------------------------

    def _probe_duration(self, path: Path) -> float:
        try:
            from mutagen.mp3 import MP3
            return MP3(str(path)).info.length
        except Exception:
            size_bits = path.stat().st_size * 8
            return size_bits / 128_000
