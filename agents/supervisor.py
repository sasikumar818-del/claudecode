"""Supervisor Agent — orchestrates and monitors all other agents in the pipeline.

The supervisor uses Claude to make decisions at each stage:
- Validates outputs before passing them to the next agent
- Retries failed steps with adjusted instructions
- Can skip or reorder steps based on context
- Provides a structured audit log of every decision
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import anthropic

from agents.image_generator import ImageGeneratorAgent
from agents.review import ReviewAgent
from agents.script_writer import ScriptWriterAgent
from agents.storyboard import StoryboardAgent
from agents.video_assembler import VideoAssemblerAgent
from agents.voiceover import VoiceoverAgent
from config.settings import Settings, get_settings
from models.production import ProductionPackage
from models.script import ScriptRequest
from models.storyboard import Storyboard


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepRecord:
    name: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    supervisor_decision: Optional[str] = None
    attempt: int = 1

    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


@dataclass
class SupervisorLog:
    pipeline_run_id: str
    topic: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    steps: list[StepRecord] = field(default_factory=list)
    final_status: StepStatus = StepStatus.PENDING

    def to_dict(self) -> dict:
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "topic": self.topic,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "final_status": self.final_status.value,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "attempt": s.attempt,
                    "duration_seconds": s.duration_seconds(),
                    "supervisor_decision": s.supervisor_decision,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


SUPERVISOR_SYSTEM_PROMPT = """You are a production supervisor for an AI video pipeline.
You receive a summary of a completed pipeline step and decide what to do next.

Respond ONLY with valid JSON matching this schema:
{
  "decision": "<proceed|retry|skip|abort>",
  "reason": "<one sentence explaining your decision>",
  "adjusted_instructions": "<optional guidance for the next step or retry, empty string if none>"
}

Decision rules:
- "proceed": output looks valid and complete, move to the next step
- "retry": output has issues but can be fixed by running the step again (max 2 retries)
- "skip": step is optional and failure is non-critical (e.g., on_screen_text overlay)
- "abort": unrecoverable error, pipeline must stop
"""


class SupervisorAgent:
    """
    Controls and monitors all pipeline agents.

    Responsibilities:
    - Instantiate and manage every specialist agent
    - Call each agent in the correct order
    - Ask Claude to validate each output and decide: proceed / retry / skip / abort
    - Record a full audit log saved to output/final/<run_id>_supervisor_log.json
    - Print a live status table to stdout
    """

    MAX_RETRIES = 2

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self.model = settings.claude_model
        self.final_dir = settings.output_subdirs["final"]
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Specialist agents
        self.script_writer = ScriptWriterAgent(settings)
        self.storyboard = StoryboardAgent(settings)
        self.voiceover = VoiceoverAgent(settings)
        self.image_generator = ImageGeneratorAgent(settings)
        self.video_assembler = VideoAssemblerAgent(settings)
        self.review = ReviewAgent(settings)

        self._log: Optional[SupervisorLog] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, request: ScriptRequest) -> ProductionPackage:
        self._log = SupervisorLog(
            pipeline_run_id="",   # filled after package is created
            topic=request.topic,
        )
        self._print_header(request)

        # Ensure all output subdirs exist
        for path in self.settings.output_subdirs.values():
            path.mkdir(parents=True, exist_ok=True)

        package: Optional[ProductionPackage] = None

        # ── Steps 1 & 2: Script + Storyboard (or bypass) ──────────────
        if request.storyboard_path and request.storyboard_path.exists():
            # Bypass Claude script/storyboard generation — load pre-defined JSON
            print(f"\n  [Supervisor] Loading pre-defined storyboard: {request.storyboard_path}")
            storyboard: Storyboard = Storyboard.from_json_file(request.storyboard_path)
            print(f"  [Supervisor] Loaded {len(storyboard.scenes)} scenes (lang={storyboard.language})")
        else:
            # ── Step 1: Script Writer ──────────────────────────────────
            script = self._run_step(
                name="Script Writer",
                fn=lambda: self.script_writer.run(request),
                summary_fn=lambda r: (
                    f"Generated script titled '{r.title}' with {len(r.sections)} sections "
                    f"({r.total_estimated_duration:.0f}s estimated)."
                ),
            )

            # ── Step 2: Storyboard ─────────────────────────────────────
            storyboard = self._run_step(
                name="Storyboard",
                fn=lambda: self.storyboard.run(script),
                summary_fn=lambda r: (
                    f"Created storyboard with {len(r.scenes)} scenes."
                ),
            )

        # ── Step 3: Voiceover ──────────────────────────────────────────
        audio_assets = self._run_step(
            name="Voiceover",
            fn=lambda: self.voiceover.run(storyboard),
            summary_fn=lambda r: (
                f"Generated {len(r)} audio files, "
                f"total {sum(a.duration_seconds for a in r):.1f}s."
            ),
        )

        # ── Step 4: Image Generator ────────────────────────────────────
        image_assets = self._run_step(
            name="Image Generator",
            fn=lambda: self.image_generator.run(storyboard),
            summary_fn=lambda r: (
                f"Generated {len(r)} images using backend '{r[0].backend}'."
            ),
        )

        # ── Step 5: Build Production Package ──────────────────────────
        package = ProductionPackage(
            storyboard=storyboard,
            audio_assets=audio_assets,
            image_assets=image_assets,
            created_at=datetime.utcnow(),
        )
        self._log.pipeline_run_id = package.pipeline_run_id

        # ── Step 6: Video Assembler ────────────────────────────────────
        package = self._run_step(
            name="Video Assembler",
            fn=lambda: self.video_assembler.run(package),
            summary_fn=lambda r: (
                f"Assembled {len(r.video_clips)} clips into final video: "
                f"{r.final_video_path}"
            ),
        )

        # ── Step 7: Review ─────────────────────────────────────────────
        package = self._run_step(
            name="Review",
            fn=lambda: self.review.run(package),
            summary_fn=lambda r: (
                f"Preview created: {r.preview_video_path}"
            ),
        )

        self._log.finished_at = datetime.utcnow()
        self._log.final_status = StepStatus.SUCCESS
        self._save_log()
        self._print_summary(package)
        return package

    # ------------------------------------------------------------------
    # Step runner with retry + supervisor validation
    # ------------------------------------------------------------------

    def _run_step(
        self,
        name: str,
        fn,
        summary_fn,
    ) -> Any:
        record = StepRecord(name=name)
        self._log.steps.append(record)

        for attempt in range(1, self.MAX_RETRIES + 2):
            record.attempt = attempt
            record.status = StepStatus.RUNNING if attempt == 1 else StepStatus.RETRYING
            record.started_at = datetime.utcnow()
            self._print_step(name, record.status, attempt)

            try:
                result = fn()
                record.finished_at = datetime.utcnow()

                # Ask supervisor whether to proceed
                summary = summary_fn(result)
                decision, reason, _ = self._supervisor_decision(name, summary, error=None)
                record.supervisor_decision = f"{decision}: {reason}"

                if decision == "proceed":
                    record.status = StepStatus.SUCCESS
                    self._print_step_result(name, StepStatus.SUCCESS, reason, record.duration_seconds())
                    return result

                if decision == "skip":
                    record.status = StepStatus.SKIPPED
                    self._print_step_result(name, StepStatus.SKIPPED, reason, record.duration_seconds())
                    return result

                if decision == "retry" and attempt <= self.MAX_RETRIES:
                    self._print_step_result(name, StepStatus.RETRYING, reason, record.duration_seconds())
                    continue

                # abort or retries exhausted
                record.status = StepStatus.FAILED
                self._log.final_status = StepStatus.FAILED
                self._save_log()
                raise RuntimeError(
                    f"Supervisor aborted pipeline at step '{name}': {reason}"
                )

            except RuntimeError:
                raise
            except Exception as exc:
                record.finished_at = datetime.utcnow()
                record.error = traceback.format_exc()
                error_summary = f"Exception: {exc}"

                decision, reason, _ = self._supervisor_decision(name, summary="", error=error_summary)
                record.supervisor_decision = f"{decision}: {reason}"

                if decision == "retry" and attempt <= self.MAX_RETRIES:
                    self._print_step_result(name, StepStatus.RETRYING, reason, record.duration_seconds())
                    continue

                if decision == "skip":
                    record.status = StepStatus.SKIPPED
                    self._print_step_result(name, StepStatus.SKIPPED, reason, record.duration_seconds())
                    return None

                record.status = StepStatus.FAILED
                self._log.final_status = StepStatus.FAILED
                self._save_log()
                raise RuntimeError(
                    f"Pipeline failed at step '{name}' after {attempt} attempt(s): {exc}"
                ) from exc

        # Should not reach here
        raise RuntimeError(f"Step '{name}' exhausted all retries.")

    # ------------------------------------------------------------------
    # Supervisor LLM call
    # ------------------------------------------------------------------

    def _supervisor_decision(
        self, step_name: str, summary: str, error: Optional[str]
    ) -> tuple[str, str, str]:
        if error:
            user_msg = (
                f"Step '{step_name}' encountered an error.\n"
                f"Error: {error}\n"
                "Decide: retry, skip, or abort?"
            )
        else:
            user_msg = (
                f"Step '{step_name}' completed.\n"
                f"Summary: {summary}\n"
                "Decide: proceed, retry, skip, or abort?"
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SUPERVISOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
            return data["decision"], data["reason"], data.get("adjusted_instructions", "")
        except Exception:
            # Fallback: if Claude returns unparseable output, proceed
            return "proceed", "Supervisor response unparseable; defaulting to proceed.", ""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _save_log(self) -> None:
        if not self._log.pipeline_run_id:
            self._log.pipeline_run_id = "unknown"
        log_path = self.final_dir / f"{self._log.pipeline_run_id}_supervisor_log.json"
        log_path.write_text(json.dumps(self._log.to_dict(), indent=2))
        print(f"\n  [Supervisor] Audit log saved → {log_path}")

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _print_header(self, request: ScriptRequest) -> None:
        print("\n" + "═" * 62)
        print("  SUPERVISOR AGENT — AI Video Production Pipeline")
        print("═" * 62)
        print(f"  Topic    : {request.topic}")
        print(f"  Duration : {request.target_duration_seconds}s  |  Tone: {request.tone}")
        print("─" * 62)

    def _print_step(self, name: str, status: StepStatus, attempt: int) -> None:
        tag = f"[attempt {attempt}]" if attempt > 1 else ""
        print(f"\n  ▶  {name} {tag}".rstrip())

    def _print_step_result(
        self, name: str, status: StepStatus, reason: str, duration: Optional[float]
    ) -> None:
        icons = {
            StepStatus.SUCCESS: "✓",
            StepStatus.SKIPPED: "↷",
            StepStatus.RETRYING: "↺",
            StepStatus.FAILED: "✗",
        }
        icon = icons.get(status, "?")
        dur = f"  ({duration:.1f}s)" if duration else ""
        print(f"  {icon}  {name}: {reason}{dur}")

    def _print_summary(self, package: ProductionPackage) -> None:
        total = sum(c.duration_seconds for c in package.video_clips)
        print("\n" + "═" * 62)
        print("  SUPERVISOR — PIPELINE COMPLETE")
        print("═" * 62)
        for step in self._log.steps:
            icon = {"success": "✓", "skipped": "↷", "failed": "✗"}.get(step.status.value, "?")
            dur = f"{step.duration_seconds():.1f}s" if step.duration_seconds() else "-"
            print(f"  {icon}  {step.name:<22} {step.status.value:<10} {dur:>8}")
        print("─" * 62)
        print(f"  Scenes   : {len(package.storyboard.scenes)}")
        print(f"  Duration : {total:.1f}s")
        print(f"  Preview  : {package.preview_video_path}")
        print(f"  Final    : {package.final_video_path}")
        print("═" * 62 + "\n")
