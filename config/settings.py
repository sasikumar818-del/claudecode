from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Anthropic (Claude) — optional; supervisor skips LLM validation when absent
    anthropic_api_key: Optional[SecretStr] = None

    # TTS backend selection
    tts_backend: Literal["espeak", "gtts", "edge-tts", "elevenlabs"] = "espeak"

    # ElevenLabs TTS — only needed when tts_backend="elevenlabs"
    elevenlabs_api_key: Optional[SecretStr] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    # Image backend selection
    image_backend: Literal["huggingface", "pil", "dalle", "stability"] = "huggingface"

    # HuggingFace Inference API — free tier; empty string = anonymous
    huggingface_api_key: str = ""
    huggingface_image_model: str = "black-forest-labs/FLUX.1-schnell"

    # OpenAI (DALL-E 3) — only needed when image_backend="dalle"
    openai_api_key: Optional[SecretStr] = None

    # Stability AI — only needed when image_backend="stability"
    stability_api_key: Optional[SecretStr] = None

    # Runway (video animation) — only needed when video_backend="runway"
    runway_api_key: Optional[SecretStr] = None
    runway_api_base: str = "https://api.dev.runwayml.com/v1"

    # Backends
    video_backend: Literal["runway", "none"] = "none"

    # Claude model
    claude_model: str = "claude-opus-4-6"

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

    def voice_id_for_language(self, language: str) -> str:
        lang_to_voice = {
            "ta": "21m00Tcm4TlvDq8ikWAM",
            "hi": "21m00Tcm4TlvDq8ikWAM",
            "en": "21m00Tcm4TlvDq8ikWAM",
        }
        return lang_to_voice.get(language, self.elevenlabs_voice_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
