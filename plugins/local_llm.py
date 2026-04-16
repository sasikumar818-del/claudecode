"""Local LLM Script Writer — uses Ollama for fully offline script generation.

Requires Ollama running locally:
    ollama serve          # start the server
    ollama pull llama3.2  # or any other model

Set in .env:
    USE_LOCAL_LLM=true
    OLLAMA_MODEL=llama3.2
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


class LocalLLMScriptWriter(AgentPlugin):
    """Replaces the Anthropic-based Script Writer with a local Ollama LLM."""

    @property
    def name(self) -> str:
        return "local_llm_script_writer"

    @property
    def replaces(self) -> str:
        return "Script Writer"

    def setup(self, settings: Any) -> None:
        self._ollama_url = f"{settings.ollama_base_url}/api/generate"
        self._model = settings.ollama_model
        self._output_dir = settings.output_subdirs["scripts"]
        self._output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  [LocalLLM] Ollama at {settings.ollama_base_url} "
            f"(model: {self._model})"
        )

    def run(self, input_data: ScriptRequest) -> VideoScript:
        import requests

        prompt = _PROMPT_TEMPLATE.format(
            topic=input_data.topic,
            duration=input_data.target_duration_seconds,
            tone=input_data.tone,
            language=input_data.language,
        )

        resp = requests.post(
            self._ollama_url,
            json={"model": self._model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json()["response"]

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
        print(f"  [LocalLLM] Script saved → {out}")
        return script
