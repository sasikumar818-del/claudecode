from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from models.script import VideoScript


class ShotType(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    AERIAL = "aerial"
    POV = "pov"
    TIMELAPSE = "timelapse"


class Scene(BaseModel):
    scene_id: int
    section_id: int
    shot_type: ShotType
    visual_description: str
    camera_direction: str
    on_screen_text: str = ""
    narration_text: str
    duration_seconds: float


class Storyboard(BaseModel):
    script: VideoScript
    scenes: list[Scene]
    created_at: datetime = Field(default_factory=datetime.utcnow)
