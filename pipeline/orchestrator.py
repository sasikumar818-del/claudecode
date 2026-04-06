"""Pipeline Orchestrator — wires all agents together in a sequential pipeline."""
from __future__ import annotations

from datetime import datetime

from agents.image_generator import ImageGeneratorAgent
from agents.review import ReviewAgent
from agents.script_writer import ScriptWriterAgent
from agents.storyboard import StoryboardAgent
from agents.video_assembler import VideoAssemblerAgent
from agents.voiceover import VoiceoverAgent
from config.settings import Settings, get_settings
from models.production import ProductionPackage
from models.script import ScriptRequest


PIPELINE_STEPS = [
    "Script Writer",
    "Storyboard",
    "Voiceover",
    "Image Generator",
    "Video Assembler",
    "Review",
]


class Pipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ensure_output_dirs()

        self.script_writer = ScriptWriterAgent(settings)
        self.storyboard = StoryboardAgent(settings)
        self.voiceover = VoiceoverAgent(settings)
        self.image_generator = ImageGeneratorAgent(settings)
        self.video_assembler = VideoAssemblerAgent(settings)
        self.review = ReviewAgent(settings)

    def run(self, request: ScriptRequest) -> ProductionPackage:
        total = len(PIPELINE_STEPS)

        self._log_step("Script Writer", 1, total)
        script = self.script_writer.run(request)

        self._log_step("Storyboard", 2, total)
        storyboard = self.storyboard.run(script)

        self._log_step("Voiceover", 3, total)
        audio_assets = self.voiceover.run(storyboard)

        self._log_step("Image Generator", 4, total)
        image_assets = self.image_generator.run(storyboard)

        self._log_step("Build Production Package", 5, total)
        package = ProductionPackage(
            storyboard=storyboard,
            audio_assets=audio_assets,
            image_assets=image_assets,
            created_at=datetime.utcnow(),
        )

        self._log_step("Video Assembler", 5, total)
        package = self.video_assembler.run(package)

        self._log_step("Review", 6, total)
        package = self.review.run(package)

        return package

    def _log_step(self, name: str, index: int, total: int) -> None:
        print(f"\n[{index}/{total}] {name} — running…")

    def _ensure_output_dirs(self) -> None:
        for path in self.settings.output_subdirs.values():
            path.mkdir(parents=True, exist_ok=True)


def run_pipeline(
    topic: str,
    duration: int = 60,
    tone: str = "educational",
    language: str = "en",
) -> ProductionPackage:
    request = ScriptRequest(
        topic=topic,
        target_duration_seconds=duration,
        tone=tone,
        language=language,
    )
    return Pipeline(get_settings()).run(request)
