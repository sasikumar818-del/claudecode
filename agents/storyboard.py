"""Storyboard Agent — uses Claude to decompose a script into visual scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anthropic

from config.settings import Settings, get_settings
from models.script import VideoScript
from models.storyboard import Scene, ShotType, Storyboard

SYSTEM_PROMPT = """You are a professional cinematographer and storyboard director.
Given a video script broken into sections, produce a detailed storyboard with one or more scenes per section.

Return ONLY valid JSON (no markdown, no extra text) matching this schema:
{
  "scenes": [
    {
      "scene_id": 1,
      "section_id": <int matching a section_id from the script>,
      "shot_type": "<wide|medium|close_up|aerial|pov|timelapse>",
      "visual_description": "<detailed image-generation prompt describing what is on screen>",
      "camera_direction": "<e.g. slow pan left, static shot, dolly zoom>",
      "on_screen_text": "<optional lower-third title or caption, empty string if none>",
      "narration_text": "<copied verbatim from the matching section narration>",
      "duration_seconds": <float>
    }
  ]
}

Rules:
- visual_description must be detailed enough to use directly as an image generation prompt.
- Include cinematic language: lighting, atmosphere, colour palette, style.
- on_screen_text should appear only at key moments (title cards, statistics, names).
- scene durations should match the section durations.
"""


class StoryboardAgent:
    def __init__(self, settings: Settings) -> None:
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self.model = settings.claude_model
        self.output_dir = settings.output_subdirs["storyboards"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, script: VideoScript) -> Storyboard:
        prompt = self._build_prompt(script)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        storyboard = self._parse_response(raw, script)

        out_file = self.output_dir / f"{script.created_at.strftime('%Y%m%d_%H%M%S')}_storyboard.json"
        out_file.write_text(storyboard.model_dump_json(indent=2))
        print(f"  [Storyboard] Saved → {out_file}")
        return storyboard

    def _build_prompt(self, script: VideoScript) -> str:
        sections_json = json.dumps(
            [s.model_dump() for s in script.sections], indent=2
        )
        return (
            f"Video title: {script.title}\n"
            f"Tone: {script.request.tone}\n\n"
            f"Script sections:\n{sections_json}\n\n"
            "Create a detailed storyboard. Return JSON as specified."
        )

    def _parse_response(self, raw: str, script: VideoScript) -> Storyboard:
        data = json.loads(raw.strip())
        scenes = [
            Scene(
                scene_id=s["scene_id"],
                section_id=s["section_id"],
                shot_type=ShotType(s["shot_type"]),
                visual_description=s["visual_description"],
                camera_direction=s["camera_direction"],
                on_screen_text=s.get("on_screen_text", ""),
                narration_text=s["narration_text"],
                duration_seconds=float(s["duration_seconds"]),
            )
            for s in data["scenes"]
        ]
        return Storyboard(script=script, scenes=scenes)


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Run the Storyboard Agent")
    parser.add_argument("--script-file", required=True, help="Path to script JSON file")
    args = parser.parse_args()

    script = VideoScript.model_validate_json(Path(args.script_file).read_text())
    result = StoryboardAgent(get_settings()).run(script)
    print(result.model_dump_json(indent=2))
