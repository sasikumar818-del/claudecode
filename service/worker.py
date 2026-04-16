"""24/7 Studio Worker — continuously dequeues and processes video jobs.

The worker runs in a tight loop, picking up pending jobs from the SQLite
queue and running them through the full pipeline (with local or cloud
plugins depending on settings).  It handles SIGTERM/SIGINT gracefully,
finishing the current job before shutting down.
"""
from __future__ import annotations

import signal
import traceback
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from config.settings import Settings
from models.script import ScriptRequest
from pipeline.orchestrator import Pipeline
from plugins.registry import PluginRegistry
from service.queue import Job, JobQueue

if TYPE_CHECKING:
    pass

# Seconds to wait between queue polls when no job is pending
_POLL_INTERVAL = 5


class StudioWorker:
    """Background worker that processes video jobs indefinitely."""

    def __init__(self, settings: Settings, queue: JobQueue) -> None:
        self.settings = settings
        self.queue = queue
        self._stop = Event()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Block and run until stop() is called or a signal is received."""
        recovered = self.queue.reset_stale()
        if recovered:
            print(f"[Worker] Re-queued {recovered} stale job(s) from previous run.")

        self._register_signals()
        print("[Worker] Studio worker started — waiting for jobs…")

        while not self._stop.is_set():
            job = self.queue.dequeue()
            if job is None:
                self._stop.wait(timeout=_POLL_INTERVAL)
                continue
            self._process(job)

        print("[Worker] Worker stopped.")

    def stop(self) -> None:
        """Signal the worker to finish the current job and exit."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _process(self, job: Job) -> None:
        print(f"\n[Worker] ▶  Job {job.id[:8]}…  topic={job.topic!r}")
        try:
            registry = self._build_registry()
            pipeline = Pipeline(self.settings, registry=registry)
            request = ScriptRequest(
                topic=job.topic,
                target_duration_seconds=job.duration,
                tone=job.tone,
                language=job.language,
            )
            package = pipeline.run(request)

            log_path = str(
                self.settings.output_subdirs["final"]
                / f"{package.pipeline_run_id}_supervisor_log.json"
            )
            self.queue.complete(
                job.id,
                final_video=str(package.final_video_path),
                preview_video=str(package.preview_video_path),
                log_path=log_path,
            )
            print(f"[Worker] ✓  Job {job.id[:8]} complete → {package.final_video_path}")

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[Worker] ✗  Job {job.id[:8]} failed: {exc}")
            self.queue.fail(job.id, tb)

    # ------------------------------------------------------------------
    # Plugin registry assembly
    # ------------------------------------------------------------------

    def _build_registry(self) -> PluginRegistry:
        """Build a fresh PluginRegistry for this job, loading local plugins
        as configured by settings flags, then any user plugins from disk."""
        registry = PluginRegistry()
        s = self.settings

        if getattr(s, "use_local_llm", False):
            from plugins.local_llm import LocalLLMScriptWriter
            registry.register(LocalLLMScriptWriter())

        if getattr(s, "use_local_tts", False):
            from plugins.local_tts import LocalTTSVoiceover
            registry.register(LocalTTSVoiceover())

        if getattr(s, "use_local_image", False):
            from plugins.local_image import LocalSDImageGenerator
            registry.register(LocalSDImageGenerator())

        if getattr(s, "use_local_video", False):
            from plugins.local_video import LocalKenBurnsAssembler
            registry.register(LocalKenBurnsAssembler())

        if getattr(s, "use_gemini_llm", False):
            from plugins.gemini_llm import GeminiScriptWriter
            registry.register(GeminiScriptWriter())

        if getattr(s, "use_imaginepro_image", False):
            from plugins.imaginepro_image import ImagineProImageGenerator
            registry.register(ImagineProImageGenerator())

        if getattr(s, "use_imaginepro_video", False):
            from plugins.imaginepro_video import ImagineProVideoAssembler
            registry.register(ImagineProVideoAssembler())

        # Load any extra user-defined plugins from the plugin directory
        registry.load_from_directory(s.plugin_dir)
        return registry

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signals(self) -> None:
        def _handler(sig, frame):
            print(f"\n[Worker] Signal {sig} — finishing current job then stopping…")
            self.stop()

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
