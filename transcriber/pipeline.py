"""The pipeline: probe -> normalize -> chunk -> transcribe (with retry + checkpoints) -> merge -> post-process.

Every intermediate artifact lives in a per-job `work_dir`:

    work_dir/
      audio.wav            normalized 16 kHz mono PCM (written atomically)
      chunks/0000.json     one checkpoint per transcribed chunk

If the process dies half-way through a 3-hour file, re-running with the same
work_dir skips normalization and every finished chunk, and resumes at the
first missing one. That is the core of the recovery story.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

from . import audio
from .engine import Engine
from .errors import PermanentError, TransientError
from .models import Chunk, Segment, Transcript, Word
from .postprocess import postprocess

log = logging.getLogger(__name__)
T = TypeVar("T")

ProgressFn = Callable[[int, int], None]  # (chunks_done, chunks_total)


@dataclass
class PipelineConfig:
    language: Optional[str] = None   # None -> auto-detect on first chunk, then pin
    chunk_seconds: float = 600.0     # 10 min: bounded memory, meaningful progress, cheap retries
    max_attempts: int = 3            # per chunk
    retry_base_delay: float = 1.0    # seconds, doubled each attempt


def with_retries(fn: Callable[[], T], attempts: int, base_delay: float, what: str) -> T:
    """Retry transient failures with exponential backoff; never retry permanent ones."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except PermanentError:
            raise
        except TransientError as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


class TranscriptionPipeline:
    def __init__(self, engine: Engine, config: Optional[PipelineConfig] = None):
        self.engine = engine
        self.config = config or PipelineConfig()

    def run(self, src: str | Path, work_dir: str | Path,
            progress: Optional[ProgressFn] = None) -> Transcript:
        cfg = self.config
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        # 1. Validate + inspect. Fails fast (PermanentError) on non-audio input.
        info = audio.probe(src)

        # 2. Normalize every format to one canonical WAV (skipped when resuming).
        wav = work / "audio.wav"
        duration = audio.wav_duration(wav) if wav.exists() else audio.normalize(src, wav)

        # 3. Plan chunks, cutting long audio at quiet points.
        chunks = audio.plan_chunks(wav, duration, cfg.chunk_seconds)
        log.info("%s: %.1fs of %s/%s audio -> %d chunk(s)",
                 Path(src).name, duration, info.format_name, info.codec, len(chunks))

        # 4. Transcribe chunk by chunk, checkpointing each one.
        language = cfg.language
        segments: list[Segment] = []
        resumed = 0
        for chunk in chunks:
            ckpt = work / "chunks" / f"{chunk.index:04d}.json"
            result = self._load_checkpoint(ckpt, chunk)
            if result is not None:
                resumed += 1
            else:
                result = with_retries(
                    lambda: self._transcribe_chunk(wav, chunk, language),
                    cfg.max_attempts, cfg.retry_base_delay, f"chunk {chunk.index}",
                )
                _atomic_write_json(ckpt, result)
            # Pin the language detected on the first chunk: consistent output
            # across chunks and skips re-running language ID every time.
            language = language or result["language"]
            segments.extend(Segment.from_dict(s) for s in result["segments"])
            if progress:
                progress(chunk.index + 1, len(chunks))

        # 5. Merge + clean.
        return Transcript(
            segments=postprocess(segments),
            language=language,
            duration=round(duration, 3),
            source=info,
            metadata={
                "engine": self.engine.name,
                "chunks": len(chunks),
                "chunks_resumed_from_checkpoint": resumed,
                "processing_seconds": round(time.perf_counter() - t0, 2),
                "real_time_factor": round((time.perf_counter() - t0) / max(duration, 1e-6), 3),
            },
        )

    def _transcribe_chunk(self, wav: Path, chunk: Chunk, language: Optional[str]) -> dict:
        samples = audio.read_window(wav, chunk.start, chunk.end)
        res = self.engine.transcribe(samples, language=language)
        # Shift chunk-relative timestamps onto the original file's timeline.
        segs = [
            Segment(
                id=-1,
                start=chunk.start + s.start,
                end=min(chunk.start + s.end, chunk.end),
                text=s.text,
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                words=[Word(chunk.start + w.start, chunk.start + w.end, w.word, w.probability)
                       for w in s.words] if s.words else None,
            )
            for s in res.segments
        ]
        return {"chunk": asdict(chunk), "language": res.language,
                "segments": [asdict(s) for s in segs]}

    @staticmethod
    def _load_checkpoint(path: Path, chunk: Chunk) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # corrupt checkpoint -> just redo that chunk
        # Only reuse if the chunk boundaries are identical (config may have changed).
        return data if data.get("chunk") == asdict(chunk) else None
