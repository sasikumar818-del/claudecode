from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Anthropic (Claude)
    anthropic_api_key: SecretStr

    # ElevenLabs TTS
    elevenlabs_api_key: SecretStr
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # default: Rachel (English)
    elevenlabs_voice_id_ta: str = ""                      # Tamil voice (set in .env)
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    # OpenAI (DALL-E 3)
    openai_api_key: SecretStr

    # Stability AI (alternative image backend)
    stability_api_key: Optional[SecretStr] = None

    # Runway (video animation)
    runway_api_key: Optional[SecretStr] = None
    runway_api_base: str = "https://api.dev.runwayml.com/v1"

    # Backends
    image_backend: Literal["dalle", "stability"] = "dalle"
    video_backend: Literal["runway", "none"] = "runway"

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
        """Return the appropriate ElevenLabs voice ID for the given ISO language code."""
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
