"""SQLite-backed persistent job queue for the Video Production Studio.

Jobs flow through these states:
  pending → running → completed
                    ↘ failed
  pending → cancelled
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Job:
    id: str
    topic: str
    duration: int
    tone: str
    language: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    final_video_path: Optional[str]
    preview_video_path: Optional[str]
    error: Optional[str]
    log_path: Optional[str]


class JobQueue:
    """Thread-safe, SQLite-backed job queue."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    _DDL = """
        CREATE TABLE IF NOT EXISTS jobs (
            id                  TEXT PRIMARY KEY,
            topic               TEXT NOT NULL,
            duration            INTEGER DEFAULT 60,
            tone                TEXT    DEFAULT 'educational',
            language            TEXT    DEFAULT 'en',
            status              TEXT    DEFAULT 'pending',
            created_at          TEXT    NOT NULL,
            started_at          TEXT,
            completed_at        TEXT,
            final_video_path    TEXT,
            preview_video_path  TEXT,
            error               TEXT,
            log_path            TEXT
        )
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(self._DDL)

    @staticmethod
    def _to_job(row: sqlite3.Row) -> Job:
        def _dt(v: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(v) if v else None

        return Job(
            id=row["id"],
            topic=row["topic"],
            duration=row["duration"],
            tone=row["tone"],
            language=row["language"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=_dt(row["started_at"]),
            completed_at=_dt(row["completed_at"]),
            final_video_path=row["final_video_path"],
            preview_video_path=row["preview_video_path"],
            error=row["error"],
            log_path=row["log_path"],
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def submit(
        self,
        topic: str,
        duration: int = 60,
        tone: str = "educational",
        language: str = "en",
    ) -> str:
        """Enqueue a new video job and return its ID."""
        job_id = uuid.uuid4().hex
        with self._conn() as c:
            c.execute(
                """INSERT INTO jobs
                   (id, topic, duration, tone, language, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (job_id, topic, duration, tone, language, datetime.utcnow().isoformat()),
            )
        return job_id

    def dequeue(self) -> Optional[Job]:
        """Atomically pop the next pending job and mark it running.

        Returns ``None`` when the queue is empty.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE status='pending' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            c.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), row["id"]),
            )
            return self._to_job(row)

    def complete(
        self,
        job_id: str,
        final_video: str,
        preview_video: str,
        log_path: str,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """UPDATE jobs
                   SET status='completed', completed_at=?,
                       final_video_path=?, preview_video_path=?, log_path=?
                   WHERE id=?""",
                (
                    datetime.utcnow().isoformat(),
                    final_video,
                    preview_video,
                    log_path,
                    job_id,
                ),
            )

    def fail(self, job_id: str, error: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='failed', completed_at=?, error=? WHERE id=?",
                (datetime.utcnow().isoformat(), error[:8192], job_id),
            )

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job.  Returns False if the job isn't cancellable."""
        with self._conn() as c:
            result = c.execute(
                "UPDATE jobs SET status='cancelled' WHERE id=? AND status='pending'",
                (job_id,),
            )
            return result.rowcount > 0

    def reset_stale(self) -> int:
        """Re-queue jobs stuck in 'running' (e.g., after a crash).

        Returns the number of jobs reset.
        """
        with self._conn() as c:
            result = c.execute(
                "UPDATE jobs SET status='pending', started_at=NULL WHERE status='running'"
            )
            return result.rowcount

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._to_job(row) if row else None

    def list_jobs(self, limit: int = 50, offset: int = 0) -> list[Job]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._to_job(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}
