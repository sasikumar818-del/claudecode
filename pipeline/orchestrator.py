"""Pipeline Orchestrator — delegates to the Supervisor Agent.

The Supervisor controls all specialist agents, validates each step with
Claude, handles retries, and maintains the full audit log.
"""
from __future__ import annotations

from agents.supervisor import SupervisorAgent
from config.settings import Settings, get_settings
from models.production import ProductionPackage
from models.script import ScriptRequest


class Pipeline:
    """Thin wrapper that hands control to the SupervisorAgent."""

    def __init__(self, settings: Settings) -> None:
        self.supervisor = SupervisorAgent(settings)

    def run(self, request: ScriptRequest) -> ProductionPackage:
        return self.supervisor.run(request)


def run_pipeline(
    topic: str,
    duration: int = 60,
    tone: str = "educational",
    language: str = "en",
    storyboard_path=None,
) -> ProductionPackage:
    from pathlib import Path
    request = ScriptRequest(
        topic=topic,
        target_duration_seconds=duration,
        tone=tone,
        language=language,
        storyboard_path=Path(storyboard_path) if storyboard_path else None,
    )
    return Pipeline(get_settings()).run(request)
