from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Anthropic (Claude) — optional when running in fully local mode
    anthropic_api_key: Optional[SecretStr] = None

    # ElevenLabs TTS — optional when using local TTS
    elevenlabs_api_key: Optional[SecretStr] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    # OpenAI (DALL-E 3) — optional when using local image gen
    openai_api_key: Optional[SecretStr] = None

    # Stability AI (alternative image backend)
    stability_api_key: Optional[SecretStr] = None

    # Runway (video animation)
    runway_api_key: Optional[SecretStr] = None
    runway_api_base: str = "https://api.dev.runwayml.com/v1"

    # Backends
    image_backend: Literal["dalle", "stability"] = "dalle"
    video_backend: Literal["runway", "none"] = "runway"

    # Claude model
    claude_model: str = "claude-opus-4-6"

    # Output
    output_dir: Path = Path("output")
    watermark_text: str = "PREVIEW — NOT FOR DISTRIBUTION"

    # FFmpeg
    ffmpeg_path: str = "ffmpeg"

    # Plugins
    plugin_dir: Path = Path("plugins")

    # ── Studio service ────────────────────────────────────────────────────────
    studio_host: str = "0.0.0.0"
    studio_port: int = 8000
    studio_db_path: Path = Path("studio.db")

    # ── Local plugin flags (set to true to use local processing) ─────────────
    use_local_llm: bool = False    # Ollama replaces Anthropic script writer
    use_local_tts: bool = False    # pyttsx3 replaces ElevenLabs voiceover
    use_local_image: bool = False  # Stable Diffusion replaces DALL-E
    use_local_video: bool = False  # Ken Burns effect replaces Runway

    # ── Ollama (local LLM) ────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ── Local TTS ─────────────────────────────────────────────────────────────
    local_tts_rate: int = 150      # words per minute (pyttsx3)

    # ── Stable Diffusion ──────────────────────────────────────────────────────
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    sd_device: str = "auto"        # "cpu" | "cuda" | "mps" | "auto"
    sd_steps: int = 20
    sd_guidance_scale: float = 7.5

    # ── Gemini (cloud LLM alternative to Ollama / Anthropic) ─────────────────
    use_gemini_llm: bool = False    # Gemini replaces Anthropic script writer
    gemini_api_key: Optional[SecretStr] = None
    gemini_model: str = "gemini-2.0-flash"

    # ── ImaginePro (cloud image + video generation) ───────────────────────────
    use_imaginepro_image: bool = False  # ImaginePro replaces image generator
    use_imaginepro_video: bool = False  # ImaginePro replaces video assembler
    imaginepro_api_key: Optional[SecretStr] = None
    imaginepro_api_base: str = "https://api.imaginepro.ai/api/v1"

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
