"""Plugin base classes for the AI Video Production Pipeline.

Three plugin types are available:

- BasePlugin: lifecycle hooks (on_step_start / on_step_end) that fire around
  every supervisor step.  Useful for logging, metrics, notifications, etc.

- AgentPlugin(BasePlugin): replaces a named built-in pipeline step with a
  custom implementation.  Set ``replaces`` to the step name you want to
  override (e.g. ``"Script Writer"``).

- BackendPlugin(BasePlugin): provides an image or video generation backend
  that can be selected at runtime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.settings import Settings


class BasePlugin(ABC):
    """Base class for all pipeline plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this plugin."""
        ...

    def setup(self, settings: "Settings") -> None:
        """Called once when the plugin is initialised before the pipeline runs."""

    def teardown(self) -> None:
        """Called once after the pipeline run completes (success or failure)."""

    def on_step_start(self, step_name: str, input_data: Any) -> None:
        """Called immediately before a pipeline step executes."""

    def on_step_end(self, step_name: str, result: Any, duration: float) -> None:
        """Called after a pipeline step completes successfully.

        Args:
            step_name: The name of the step that just finished.
            result: The value returned by the step.
            duration: Wall-clock seconds the step took.
        """


class AgentPlugin(BasePlugin, ABC):
    """Plugin that replaces a named built-in pipeline step.

    Subclass this and set ``replaces`` to the exact step name shown in the
    supervisor (e.g. ``"Script Writer"``, ``"Storyboard"``, ``"Voiceover"``,
    ``"Image Generator"``, ``"Video Assembler"``, ``"Review"``).
    """

    @property
    @abstractmethod
    def replaces(self) -> str:
        """Name of the built-in pipeline step this plugin replaces."""
        ...

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the agent logic and return the step result."""
        ...


class BackendPlugin(BasePlugin, ABC):
    """Plugin that provides an image or video generation backend."""

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Type of backend: ``"image"`` or ``"video"``."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> bytes:
        """Generate content from *prompt* and return raw bytes."""
        ...
