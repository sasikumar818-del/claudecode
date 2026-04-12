"""Pipeline Orchestrator — delegates to the Supervisor Agent.

The Supervisor controls all specialist agents, validates each step with
Claude, handles retries, and maintains the full audit log.
"""
from __future__ import annotations

from typing import Optional

from agents.supervisor import SupervisorAgent
from config.settings import Settings, get_settings
from models.production import ProductionPackage
from models.script import ScriptRequest
from plugins.registry import PluginRegistry


class Pipeline:
    """Thin wrapper that hands control to the SupervisorAgent."""

    def __init__(
        self,
        settings: Settings,
        registry: Optional[PluginRegistry] = None,
    ) -> None:
        self.supervisor = SupervisorAgent(settings, registry=registry)

    def run(self, request: ScriptRequest) -> ProductionPackage:
        return self.supervisor.run(request)


def run_pipeline(
    topic: str,
    duration: int = 60,
    tone: str = "educational",
    language: str = "en",
) -> ProductionPackage:
    settings = get_settings()
    registry = PluginRegistry()
    registry.load_from_directory(settings.plugin_dir)
    request = ScriptRequest(
        topic=topic,
        target_duration_seconds=duration,
        tone=tone,
        language=language,
    )
    return Pipeline(settings, registry=registry).run(request)
