"""REST API.

    POST /v1/transcriptions                  upload audio -> 202 + job id
    GET  /v1/transcriptions/{id}             status, progress, and result when done
    GET  /v1/transcriptions/{id}/transcript  ?format=json|srt|vtt|txt
    GET  /healthz

Transcription is asynchronous on purpose: a long file can take minutes, which
is longer than most load balancers / clients will hold an HTTP request open.
"""
from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from . import audio
from .config import Settings
from .engine import Engine, WhisperEngine
from .errors import UnsupportedAudioError
from .jobs import Job, JobManager, QueueFullError, Status, default_pipeline_factory
from .postprocess import FORMATTERS, MEDIA_TYPES

log = logging.getLogger(__name__)
_COPY_BUF = 1024 * 1024


def create_app(settings: Optional[Settings] = None, engine: Optional[Engine] = None) -> FastAPI:
    settings = settings or Settings()
    engine = engine or WhisperEngine(settings.model, settings.device, settings.compute_type,
                                     num_workers=settings.workers)
    manager = JobManager(settings.data_dir,
                         default_pipeline_factory(engine, settings.chunk_seconds),
                         workers=settings.workers, max_queue=settings.max_queue,
                         max_attempts=settings.max_job_attempts)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.recover()
        yield
        manager.shutdown()

    app = FastAPI(title="Transcription Service", version="1.0.0", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "engine": engine.name}

    @app.post("/v1/transcriptions", status_code=202)
    def create_transcription(request: Request,
                             file: UploadFile = File(...),
                             language: Optional[str] = Form(None)):
        # Sync handler -> FastAPI runs it in a threadpool, so the blocking
        # disk copy + ffprobe below never stall the event loop.
        job_id, job_dir = manager.new_job_dir()
        suffix = Path(file.filename or "").suffix.lower()[:10]
        input_path = job_dir / f"input{suffix}"
        limit = settings.max_upload_mb * 1024 * 1024

        try:
            # Stream to disk in 1 MB pieces: never hold a whole upload in memory.
            written = 0
            with open(input_path, "wb") as out:
                while buf := file.file.read(_COPY_BUF):
                    written += len(buf)
                    if written > limit:
                        raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
                    out.write(buf)
            # Reject non-audio immediately rather than after it sits in the queue.
            info = audio.probe(input_path)
        except UnsupportedAudioError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(415, str(exc))
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        job = Job(id=job_id, filename=file.filename or "upload",
                  input_path=str(input_path), language=language)
        try:
            manager.submit(job)
        except QueueFullError:
            shutil.rmtree(job_dir, ignore_errors=True)
            return JSONResponse({"detail": "server busy, retry later"}, status_code=429,
                                headers={"Retry-After": "30"})

        return {**job.public(), "duration": info.duration,
                "status_url": str(request.url_for("get_transcription", job_id=job_id))}

    @app.get("/v1/transcriptions/{job_id}", name="get_transcription")
    def get_transcription(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        body = job.public()
        if job.status == Status.COMPLETED:
            body["result"] = manager.transcript(job_id).to_dict()
        return body

    @app.get("/v1/transcriptions/{job_id}/transcript")
    def get_transcript(job_id: str, format: str = Query("json", pattern="^(json|srt|vtt|txt)$")):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status != Status.COMPLETED:
            raise HTTPException(409, f"job is {job.status.value}")
        return Response(FORMATTERS[format](manager.transcript(job_id)),
                        media_type=MEDIA_TYPES[format])

    return app
