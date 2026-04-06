"""Script Writer Agent — uses Claude to generate a structured video script."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import anthropic

from config.settings import Settings, get_settings
from models.script import ScriptRequest, ScriptSection, VideoScript

SYSTEM_PROMPT = """You are a professional video scriptwriter. Given a topic, tone, and target duration,
you produce a structured video script broken into clear sections.

Return ONLY valid JSON (no markdown, no extra text) matching this schema:
{
  "title": "<video title>",
  "sections": [
    {
      "section_id": 1,
      "title": "<section title>",
      "narration_text": "<spoken narration for this section>",
      "approximate_duration_seconds": <float>
    }
  ],
  "total_estimated_duration": <float>
}

Rules:
- Keep narration natural and conversational for voiceover.
- Sections should sum close to the target duration.
- Minimum 3 sections, maximum 10.
"""


class ScriptWriterAgent:
    def __init__(self, settings: Settings) -> None:
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self.model = settings.claude_model
        self.output_dir = settings.output_subdirs["scripts"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, request: ScriptRequest) -> VideoScript:
        prompt = self._build_prompt(request)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        script = self._parse_response(raw, request)

        out_file = self.output_dir / f"{script.created_at.strftime('%Y%m%d_%H%M%S')}_script.json"
        out_file.write_text(script.model_dump_json(indent=2))
        print(f"  [ScriptWriter] Saved → {out_file}")
        return script

    def _build_prompt(self, request: ScriptRequest) -> str:
        return (
            f"Write a {request.tone} video script about: {request.topic}.\n"
            f"Target duration: {request.target_duration_seconds} seconds.\n"
            f"Language: {request.language}.\n"
            "Return the JSON structure as specified."
        )

    def _parse_response(self, raw: str, request: ScriptRequest) -> VideoScript:
        data = json.loads(raw.strip())
        sections = [ScriptSection(**s) for s in data["sections"]]
        return VideoScript(
            request=request,
            title=data["title"],
            sections=sections,
            total_estimated_duration=data["total_estimated_duration"],
            raw_llm_response=raw,
            created_at=datetime.utcnow(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Script Writer Agent")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--tone", default="educational",
                        choices=["educational", "cinematic", "promotional"])
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    req = ScriptRequest(
        topic=args.topic,
        target_duration_seconds=args.duration,
        tone=args.tone,
        language=args.language,
    )
    result = ScriptWriterAgent(get_settings()).run(req)
    print(result.model_dump_json(indent=2))
