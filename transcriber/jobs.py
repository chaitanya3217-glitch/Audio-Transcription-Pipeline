"""A tiny durable job queue: job state on disk, work on a bounded thread pool.

This is the single-node stand-in for the production design in docs/DESIGN.md
(object storage + Postgres + a real queue such as SQS). The *shape* is the
same, so the scale-out story is a swap of backends, not a rewrite:

    data/jobs/<job_id>/
      job.json         status record (written atomically)
      input<ext>       the original upload (never modified)
      work/            normalized audio + per-chunk checkpoints
      transcript.json  final result
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from .errors import PermanentError
from .models import Transcript
from .pipeline import PipelineConfig, TranscriptionPipeline

log = logging.getLogger(__name__)


class Status(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    filename: str
    input_path: str
    language: Optional[str] = None
    status: Status = Status.QUEUED
    attempts: int = 0
    progress: float = 0.0
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("input_path")
        d["status"] = self.status.value
        return d


class QueueFullError(Exception):
    pass


class JobManager:
    def __init__(self, data_dir: Path, pipeline_factory, workers: int = 2,
                 max_queue: int = 100, max_attempts: int = 3):
        self.root = Path(data_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._pipeline_factory = pipeline_factory  # (language) -> TranscriptionPipeline
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="transcribe")
        self._lock = threading.Lock()
        self._pending = 0
        self._cache: dict[str, Job] = {}
        self.max_queue = max_queue
        self.max_attempts = max_attempts

    # ---- persistence ------------------------------------------------------ #
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        """Persist to disk (durability) and to the in-memory cache (reads).

        Status polls are served from memory so readers never hold job.json
        open while a worker is replacing it (which fails on Windows).
        """
        job.updated_at = _now()
        with self._lock:
            self._cache[job.id] = replace(job)
        d = self.job_dir(job.id)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({**asdict(job), "status": job.status.value}, f)
        for attempt in range(5):  # AV scanners / indexers can briefly lock files on Windows
            try:
                os.replace(tmp, d / "job.json")
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            if job_id in self._cache:
                return replace(self._cache[job_id])
        try:
            uuid.UUID(job_id)  # also blocks path traversal via the id
            data = json.loads((self.job_dir(job_id) / "job.json").read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        data["status"] = Status(data["status"])
        return Job(**data)

    def transcript(self, job_id: str) -> Optional[Transcript]:
        path = self.job_dir(job_id) / "transcript.json"
        if not path.exists():
            return None
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ---- lifecycle -------------------------------------------------------- #
    def new_job_dir(self) -> tuple[str, Path]:
        job_id = str(uuid.uuid4())
        d = self.job_dir(job_id)
        d.mkdir(parents=True)
        return job_id, d

    def submit(self, job: Job) -> Job:
        with self._lock:
            if self._pending >= self.max_queue:
                raise QueueFullError()
            self._pending += 1
        self._save(job)
        self._pool.submit(self._run, job.id)
        return job

    def recover(self) -> int:
        """On startup, re-enqueue jobs that were queued or mid-flight when we died.

        Their chunk checkpoints survive, so PROCESSING jobs resume where they stopped.
        """
        count = 0
        for job_file in self.root.glob("*/job.json"):
            job = self.get(job_file.parent.name)
            if job and job.status in (Status.QUEUED, Status.PROCESSING):
                job.status = Status.QUEUED
                with self._lock:
                    self._pending += 1
                self._save(job)
                self._pool.submit(self._run, job.id)
                count += 1
        if count:
            log.info("recovered %d unfinished job(s)", count)
        return count

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        try:
            if job is None:
                return
            while job.attempts < self.max_attempts:
                job.attempts += 1
                job.status, job.error = Status.PROCESSING, None
                self._save(job)
                try:
                    def on_progress(done: int, total: int) -> None:
                        job.progress = round(done / total, 3)
                        self._save(job)

                    pipeline: TranscriptionPipeline = self._pipeline_factory(job.language)
                    result = pipeline.run(job.input_path, self.job_dir(job.id) / "work", on_progress)
                    (self.job_dir(job.id) / "transcript.json").write_text(
                        json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
                    job.status, job.progress = Status.COMPLETED, 1.0
                    self._save(job)
                    return
                except PermanentError as exc:
                    job.error = f"{type(exc).__name__}: {exc}"
                    break  # retrying cannot help
                except Exception as exc:  # job-level retry; chunk checkpoints make it cheap
                    log.exception("job %s attempt %d failed", job.id, job.attempts)
                    job.error = f"{type(exc).__name__}: {exc}"
            job.status = Status.FAILED
            self._save(job)
        finally:
            with self._lock:
                self._pending -= 1

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def default_pipeline_factory(engine, chunk_seconds: float):
    def factory(language: Optional[str]) -> TranscriptionPipeline:
        return TranscriptionPipeline(engine, PipelineConfig(language=language,
                                                            chunk_seconds=chunk_seconds))
    return factory
