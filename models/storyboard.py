from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

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
    section_id: int = 0
    shot_type: ShotType
    visual_description: str
    camera_direction: str = ""
    on_screen_text: str = ""
    narration_text: str
    duration_seconds: float = 30.0


class Storyboard(BaseModel):
    script: Optional[VideoScript] = None
    scenes: list[Scene]
    title: str = ""
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_json_file(cls, path: Path) -> "Storyboard":
        """Load a pre-defined storyboard from a JSON file, bypassing Claude generation."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scenes = [Scene(**s) for s in data["scenes"]]
        return cls(
            scenes=scenes,
            title=data.get("title", ""),
            language=data.get("language", "en"),
        )
