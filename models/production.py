from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from models.storyboard import Storyboard


class AudioAsset(BaseModel):
    scene_id: int
    file_path: Path
    duration_seconds: float
    voice_id: str


class ImageAsset(BaseModel):
    scene_id: int
    file_path: Path
    prompt_used: str
    backend: str
    revised_prompt: Optional[str] = None


class VideoClip(BaseModel):
    scene_id: int
    file_path: Path
    duration_seconds: float
    source: str  # "runway" | "static"


class ProductionPackage(BaseModel):
    storyboard: Storyboard
    audio_assets: list[AudioAsset] = Field(default_factory=list)
    image_assets: list[ImageAsset] = Field(default_factory=list)
    video_clips: list[VideoClip] = Field(default_factory=list)
    final_video_path: Optional[Path] = None
    preview_video_path: Optional[Path] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    pipeline_run_id: str = Field(default_factory=lambda: uuid4().hex)
