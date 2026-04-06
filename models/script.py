from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ScriptRequest(BaseModel):
    topic: str
    target_duration_seconds: int = 60
    tone: Literal["educational", "cinematic", "promotional"] = "educational"
    language: str = "en"


class ScriptSection(BaseModel):
    section_id: int
    title: str
    narration_text: str
    approximate_duration_seconds: float


class VideoScript(BaseModel):
    request: ScriptRequest
    title: str
    sections: list[ScriptSection]
    total_estimated_duration: float
    raw_llm_response: str = Field(default="", exclude=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
