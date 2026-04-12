"""Video Production Studio — 24/7 local AI video generation service.

Quick start (fully local mode):
    # 1. Copy and fill in the example env file
    cp .env.studio.example .env

    # 2. Start Ollama and pull a model (for local LLM)
    ollama serve &
    ollama pull llama3.2

    # 3. Start the studio (API server + background worker)
    python studio.py start

    # 4. Submit your first job (in another terminal or via the API)
    python studio.py submit "The history of artificial intelligence" --duration 90

    # 5. Check status
    python studio.py jobs
    python studio.py status <job_id>

    # 6. Download via API
    curl http://localhost:8000/jobs/<job_id>/video -o video.mp4

API endpoints (when running):
    GET  http://localhost:8000/            Info + job counts
    POST http://localhost:8000/jobs        Submit a job
    GET  http://localhost:8000/jobs        List jobs
    GET  http://localhost:8000/jobs/{id}   Job details
    GET  http://localhost:8000/jobs/{id}/video    Download MP4
    GET  http://localhost:8000/jobs/{id}/preview  Download preview
"""
from __future__ import annotations

import argparse
import sys
import threading

from config.settings import get_settings
from service.api import create_app
from service.queue import JobQueue
from service.worker import StudioWorker


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_start(args) -> None:
    """Start the API server and background worker (blocks until Ctrl-C)."""
    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn not installed.  Run: pip install uvicorn[standard]")

    settings = get_settings()
    queue = JobQueue(settings.studio_db_path)
    worker = StudioWorker(settings, queue)
    app = create_app(queue)

    # ── Worker daemon thread ──────────────────────────────────────────────────
    worker_thread = threading.Thread(
        target=worker.start, name="StudioWorker", daemon=True
    )
    worker_thread.start()

    _print_banner(settings)

    try:
        uvicorn.run(
            app,
            host=settings.studio_host,
            port=settings.studio_port,
            log_level="info",
        )
    finally:
        worker.stop()
        worker_thread.join(timeout=5)


def cmd_submit(args) -> None:
    """Submit a new video job and print its ID."""
    settings = get_settings()
    queue = JobQueue(settings.studio_db_path)
    job_id = queue.submit(
        topic=args.topic,
        duration=args.duration,
        tone=args.tone,
        language=args.language,
    )
    print(f"Job submitted : {job_id}")
    print(f"Check status  : python studio.py status {job_id}")
    print(f"API status    : GET http://{settings.studio_host}:{settings.studio_port}/jobs/{job_id}")


def cmd_jobs(args) -> None:
    """Print a table of recent jobs."""
    settings = get_settings()
    queue = JobQueue(settings.studio_db_path)
    jobs = queue.list_jobs(limit=30)
    if not jobs:
        print("No jobs found.")
        return

    print(f"\n{'ID':34} {'STATUS':12} {'TONE':14} {'DUR':6} {'TOPIC'}")
    print("─" * 110)
    for j in jobs:
        topic = j.topic[:55] + "…" if len(j.topic) > 56 else j.topic
        print(
            f"{j.id}  {j.status:<12} {j.tone:<14} {j.duration:>4}s  {topic}"
        )


def cmd_status(args) -> None:
    """Print detailed status of a single job."""
    settings = get_settings()
    queue = JobQueue(settings.studio_db_path)
    job = queue.get(args.job_id)
    if not job:
        print(f"Job not found: {args.job_id}")
        sys.exit(1)

    print()
    print(f"  Job ID    : {job.id}")
    print(f"  Topic     : {job.topic}")
    print(f"  Status    : {job.status}")
    print(f"  Duration  : {job.duration}s  |  Tone: {job.tone}  |  Lang: {job.language}")
    print(f"  Created   : {job.created_at}")
    if job.started_at:
        print(f"  Started   : {job.started_at}")
    if job.completed_at and job.started_at:
        elapsed = (job.completed_at - job.started_at).total_seconds()
        print(f"  Completed : {job.completed_at}  (took {elapsed:.0f}s)")
    if job.final_video_path:
        print(f"  Video     : {job.final_video_path}")
    if job.preview_video_path:
        print(f"  Preview   : {job.preview_video_path}")
    if job.error:
        print(f"\n  Error:\n{job.error[:800]}")
    print()


def cmd_cancel(args) -> None:
    settings = get_settings()
    queue = JobQueue(settings.studio_db_path)
    if queue.cancel(args.job_id):
        print(f"Job {args.job_id} cancelled.")
    else:
        print(f"Could not cancel job {args.job_id} (not in pending state).")
        sys.exit(1)


# ── CLI parser ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="studio",
        description="Video Production Studio — 24/7 local AI video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    sub.add_parser("start", help="Start the API server and worker daemon")

    # submit
    p_submit = sub.add_parser("submit", help="Submit a new video generation job")
    p_submit.add_argument("topic", help="Video topic or description")
    p_submit.add_argument("--duration", type=int, default=60,
                          help="Target duration in seconds (default: 60)")
    p_submit.add_argument("--tone", default="educational",
                          choices=["educational", "cinematic", "promotional"])
    p_submit.add_argument("--language", default="en")

    # jobs
    sub.add_parser("jobs", help="List recent jobs")

    # status
    p_status = sub.add_parser("status", help="Show details for a specific job")
    p_status.add_argument("job_id")

    # cancel
    p_cancel = sub.add_parser("cancel", help="Cancel a pending job")
    p_cancel.add_argument("job_id")

    args = parser.parse_args()
    {
        "start": cmd_start,
        "submit": cmd_submit,
        "jobs": cmd_jobs,
        "status": cmd_status,
        "cancel": cmd_cancel,
    }[args.command](args)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_banner(settings) -> None:
    w = 62
    local_flags = [
        ("LLM", settings.use_local_llm, "Ollama", "Anthropic Claude"),
        ("TTS", settings.use_local_tts, "pyttsx3", "ElevenLabs"),
        ("Image", settings.use_local_image, "Stable Diffusion", "DALL-E"),
        ("Video", settings.use_local_video, "Ken Burns", "Runway"),
    ]
    print("\n" + "═" * w)
    print("  VIDEO PRODUCTION STUDIO — 24/7 Local Service")
    print("═" * w)
    print(f"  API     : http://{settings.studio_host}:{settings.studio_port}")
    print(f"  DB      : {settings.studio_db_path}")
    print(f"  Output  : {settings.output_dir}")
    print("─" * w)
    for label, is_local, local_name, cloud_name in local_flags:
        provider = local_name if is_local else cloud_name
        mode = "LOCAL" if is_local else "cloud"
        print(f"  {label:<6}  [{mode}] {provider}")
    print("═" * w + "\n")


if __name__ == "__main__":
    main()
