"""Speech-to-text engines behind one small interface.

The pipeline depends only on `Engine.transcribe(audio, language)`, so swapping
faster-whisper for a cloud API (Deepgram, AssemblyAI, OpenAI, ...) or a GPU
server means writing one new class -- nothing else changes. `MockEngine`
uses the same seam to make tests fast and deterministic.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from .audio import SAMPLE_RATE
from .errors import TransientError
from .models import Word


@dataclass
class RawSegment:
    """A segment with timestamps relative to the audio the engine was given."""
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    words: Optional[list[Word]] = None


@dataclass
class EngineResult:
    segments: list[RawSegment]
    language: Optional[str]
    language_probability: Optional[float] = None


class Engine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> EngineResult: ...


class WhisperEngine:
    """faster-whisper (CTranslate2 port of OpenAI Whisper).

    Chosen because it is open source, runs offline, is ~4x faster than the
    reference implementation, and int8 quantization makes CPU-only use practical.
    The model is loaded lazily and once per process (it is the expensive part).
    """

    def __init__(self, model_size: str = "base", device: str = "auto",
                 compute_type: str = "int8", beam_size: int = 5,
                 vad_filter: bool = True, word_timestamps: bool = False,
                 cpu_threads: int = 0, num_workers: int = 1):
        self.name = f"faster-whisper:{model_size}"
        self._kwargs = dict(device=device, compute_type=compute_type,
                            cpu_threads=cpu_threads, num_workers=num_workers)
        self.model_size = model_size
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_size, **self._kwargs)
            return self._model

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> EngineResult:
        try:
            segments, info = self._get_model().transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,          # skip silence -> fewer hallucinations, faster
                word_timestamps=self.word_timestamps,
                condition_on_previous_text=False,     # stops repetition loops propagating
            )
            out = [
                RawSegment(
                    start=s.start, end=s.end, text=s.text,
                    avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
                    words=[Word(w.start, w.end, w.word, w.probability) for w in s.words]
                    if s.words else None,
                )
                for s in segments  # generator: decoding actually happens here
            ]
        except (RuntimeError, MemoryError) as exc:
            # CTranslate2 surfaces OOM / device errors as RuntimeError -> retryable
            raise TransientError(f"engine failure: {exc}") from exc
        return EngineResult(out, info.language, info.language_probability)


@dataclass
class MockEngine:
    """Deterministic fake: one segment per `segment_s` seconds of non-silent audio.

    `fail_times` makes the first N calls raise TransientError, for retry tests.
    """
    segment_s: float = 5.0
    fail_times: int = 0
    name: str = "mock"
    calls: int = field(default=0, init=False)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> EngineResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError(f"simulated failure #{self.calls}")
        duration = len(audio) / SAMPLE_RATE
        segs, t, i = [], 0.0, 0
        while t < duration:
            end = min(t + self.segment_s, duration)
            window = audio[int(t * SAMPLE_RATE): int(end * SAMPLE_RATE)]
            if window.size and float(np.abs(window).max()) > 1e-3:
                segs.append(RawSegment(t, end, f" mock segment {i}.", -0.2, 0.01))
            t, i = end, i + 1
        return EngineResult(segs, language or "en", 1.0)
