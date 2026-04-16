"""FastAPI REST API for the Video Production Studio.

Endpoints
---------
GET  /                      Service info and job counts
GET  /health                Liveness probe
POST /jobs                  Submit a new video job
GET  /jobs                  List jobs (newest first)
GET  /jobs/{id}             Get job details
DELETE /jobs/{id}           Cancel a pending job
GET  /jobs/{id}/video       Stream / download the final MP4
GET  /jobs/{id}/preview     Stream / download the watermarked preview MP4
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from service.queue import Job, JobQueue


# ── Request / response schemas ────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500,
                       description="What the video should be about")
    duration: int = Field(default=60, ge=10, le=600,
                          description="Target video length in seconds")
    tone: str = Field(default="educational",
                      pattern="^(educational|cinematic|promotional)$")
    language: str = Field(default="en", max_length=10,
                          description="ISO 639-1 language code")


class JobOut(BaseModel):
    id: str
    topic: str
    duration: int
    tone: str
    language: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    final_video_path: Optional[str] = None
    preview_video_path: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_job(cls, j: Job) -> "JobOut":
        def _fmt(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None

        return cls(
            id=j.id,
            topic=j.topic,
            duration=j.duration,
            tone=j.tone,
            language=j.language,
            status=j.status,
            created_at=j.created_at.isoformat(),
            started_at=_fmt(j.started_at),
            completed_at=_fmt(j.completed_at),
            final_video_path=j.final_video_path,
            preview_video_path=j.preview_video_path,
            error=j.error,
        )


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(queue: JobQueue) -> FastAPI:
    app = FastAPI(
        title="Video Production Studio",
        description="24/7 local AI video generation — submit a topic, get an MP4.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Info / health ─────────────────────────────────────────────────────────

    @app.get("/", tags=["info"])
    def index():
        return {
            "service": "Video Production Studio",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "jobs": queue.counts(),
            "endpoints": {
                "submit": "POST /jobs",
                "list": "GET /jobs",
                "status": "GET /jobs/{id}",
                "download": "GET /jobs/{id}/video",
                "preview": "GET /jobs/{id}/preview",
                "cancel": "DELETE /jobs/{id}",
            },
        }

    @app.get("/health", tags=["info"])
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    # ── Job CRUD ──────────────────────────────────────────────────────────────

    @app.post("/jobs", status_code=201, tags=["jobs"])
    def submit_job(req: SubmitRequest):
        """Submit a new video generation job and return its ID."""
        job_id = queue.submit(
            topic=req.topic,
            duration=req.duration,
            tone=req.tone,
            language=req.language,
        )
        return {
            "job_id": job_id,
            "status": "pending",
            "poll": f"/jobs/{job_id}",
        }

    @app.get("/jobs", tags=["jobs"])
    def list_jobs(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        jobs = queue.list_jobs(limit=limit, offset=offset)
        return {"jobs": [JobOut.from_job(j).model_dump() for j in jobs]}

    @app.get("/jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str):
        job = queue.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return JobOut.from_job(job)

    @app.delete("/jobs/{job_id}", tags=["jobs"])
    def cancel_job(job_id: str):
        if not queue.cancel(job_id):
            raise HTTPException(409, "Job cannot be cancelled (not in pending state)")
        return {"job_id": job_id, "status": "cancelled"}

    # ── Downloads ─────────────────────────────────────────────────────────────

    @app.get("/jobs/{job_id}/video", tags=["downloads"])
    def download_video(job_id: str):
        """Download the final (full-quality) MP4."""
        return _serve_file(queue, job_id, "final_video_path", f"{job_id}_final.mp4")

    @app.get("/jobs/{job_id}/preview", tags=["downloads"])
    def download_preview(job_id: str):
        """Download the watermarked preview MP4."""
        return _serve_file(queue, job_id, "preview_video_path", f"{job_id}_preview.mp4")

    return app


def _serve_file(queue: JobQueue, job_id: str, attr: str, filename: str) -> FileResponse:
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, f"Video not ready — job status is '{job.status}'")
    video_path = getattr(job, attr)
    if not video_path:
        raise HTTPException(409, "Video path not recorded")
    path = Path(video_path)
    if not path.exists():
        raise HTTPException(404, "Video file missing from disk")
    return FileResponse(path, media_type="video/mp4", filename=filename)
