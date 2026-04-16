from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Anthropic (Claude) — required for supervisor decisions
    anthropic_api_key: SecretStr

    # ── TTS backend ──────────────────────────────────────────────────────
    # "espeak"     — espeak-ng offline TTS (default, free, no internet needed)
    #                Install: sudo apt install espeak-ng ffmpeg
    # "gtts"       — Google TTS (free, needs internet, no key)
    # "edge-tts"   — Microsoft Edge TTS (free, needs internet, high quality)
    # "elevenlabs" — ElevenLabs paid API
    tts_backend: Literal["espeak", "gtts", "edge-tts", "elevenlabs"] = "espeak"

    # ElevenLabs (only needed when TTS_BACKEND=elevenlabs)
    elevenlabs_api_key: Optional[SecretStr] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (English)
    elevenlabs_voice_id_ta: str = ""                      # Tamil voice
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    # ── Image backend ─────────────────────────────────────────────────────
    # "huggingface" — HuggingFace Inference API, free (default)
    # "dalle"       — OpenAI DALL-E 3 (paid)
    # "stability"   — Stability AI (paid)
    image_backend: Literal["huggingface", "dalle", "stability"] = "huggingface"

    # HuggingFace (free; token optional but increases rate limits)
    huggingface_api_key: str = ""
    huggingface_image_model: str = "black-forest-labs/FLUX.1-schnell"

    # OpenAI DALL-E 3 (only needed when IMAGE_BACKEND=dalle)
    openai_api_key: Optional[SecretStr] = None

    # Stability AI (only needed when IMAGE_BACKEND=stability)
    stability_api_key: Optional[SecretStr] = None

    # ── Video backend ─────────────────────────────────────────────────────
    # "none"   — static images + MoviePy only (free, default)
    # "runway" — Runway Gen-3 animation (paid)
    video_backend: Literal["runway", "none"] = "none"

    # Runway (only needed when VIDEO_BACKEND=runway)
    runway_api_key: Optional[SecretStr] = None
    runway_api_base: str = "https://api.dev.runwayml.com/v1"

    # Claude model
    claude_model: str = "claude-opus-4-5"

    # Output
    output_dir: Path = Path("output")
    watermark_text: str = "PREVIEW — NOT FOR DISTRIBUTION"

    # Video resolution
    video_width: int = 1920
    video_height: int = 1080

    # FFmpeg
    ffmpeg_path: str = "ffmpeg"

    model_config = SettingsConfigDict(
        env_file=".env",
        secrets_strip_whitespace=True,
        extra="ignore",
    )

    def voice_id_for_language(self, language: str) -> str:
        """Return the ElevenLabs voice ID for the given ISO language code."""
        if language == "ta" and self.elevenlabs_voice_id_ta:
            return self.elevenlabs_voice_id_ta
        return self.elevenlabs_voice_id

    @property
    def output_subdirs(self) -> dict[str, Path]:
        return {
            "scripts": self.output_dir / "scripts",
            "storyboards": self.output_dir / "storyboards",
            "audio": self.output_dir / "audio",
            "images": self.output_dir / "images",
            "clips": self.output_dir / "clips",
            "final": self.output_dir / "final",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
