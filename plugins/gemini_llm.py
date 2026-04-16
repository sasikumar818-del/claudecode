"""Gemini Script Writer — uses Google Gemini API for cloud-based script generation.

Install dependency:
    pip install google-generativeai

Set in .env:
    USE_GEMINI_LLM=true
    GEMINI_API_KEY=your_api_key_here
    GEMINI_MODEL=gemini-2.0-flash   # or gemini-1.5-pro, etc.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from plugins.base import AgentPlugin
from models.script import ScriptRequest, ScriptSection, VideoScript


_PROMPT_TEMPLATE = """\
You are a professional video scriptwriter. Generate a structured video script as JSON.

Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "title": "<video title>",
  "sections": [
    {{
      "section_id": 1,
      "title": "<section title>",
      "narration_text": "<spoken narration for this section>",
      "approximate_duration_seconds": <float>
    }}
  ],
  "total_estimated_duration": <float>
}}

Rules:
- Narration must be natural and conversational (it will be spoken aloud).
- Sections should sum close to {duration} seconds total.
- Minimum 3 sections, maximum 10.
- Tone: {tone}. Language: {language}.

Topic: {topic}"""


class GeminiScriptWriter(AgentPlugin):
    """Replaces the Anthropic-based Script Writer with Google Gemini."""

    @property
    def name(self) -> str:
        return "gemini_script_writer"

    @property
    def replaces(self) -> str:
        return "Script Writer"

    def setup(self, settings: Any) -> None:
        import google.generativeai as genai

        api_key = settings.gemini_api_key
        if api_key is None:
            raise ValueError("GEMINI_API_KEY must be set when USE_GEMINI_LLM=true")
        genai.configure(api_key=api_key.get_secret_value())
        self._model = genai.GenerativeModel(settings.gemini_model)
        self._output_dir = settings.output_subdirs["scripts"]
        self._output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Gemini] Model: {settings.gemini_model}")

    def run(self, input_data: ScriptRequest) -> VideoScript:
        prompt = _PROMPT_TEMPLATE.format(
            topic=input_data.topic,
            duration=input_data.target_duration_seconds,
            tone=input_data.tone,
            language=input_data.language,
        )

        response = self._model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown fences if present (```json ... ``` or ``` ... ```)
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:].lstrip("\n")

        data = json.loads(raw)
        sections = [ScriptSection(**s) for s in data["sections"]]
        script = VideoScript(
            request=input_data,
            title=data["title"],
            sections=sections,
            total_estimated_duration=float(data["total_estimated_duration"]),
            raw_llm_response=raw,
            created_at=datetime.utcnow(),
        )

        out = self._output_dir / f"{script.created_at.strftime('%Y%m%d_%H%M%S')}_script.json"
        out.write_text(script.model_dump_json(indent=2))
        print(f"  [Gemini] Script saved → {out}")
        return script
