"""Plain data structures shared by every layer of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    start: float  # seconds from the start of the ORIGINAL file
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    words: Optional[list[Word]] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        words = [Word(**w) for w in d["words"]] if d.get("words") else None
        return cls(**{**d, "words": words})


@dataclass
class AudioInfo:
    """What ffprobe tells us about the uploaded file, before normalization."""
    format_name: str
    codec: str
    sample_rate: int
    channels: int
    duration: float
    size_bytes: int


@dataclass
class Chunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Transcript:
    segments: list[Segment]
    language: Optional[str]
    duration: float
    source: Optional[AudioInfo] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["text"] = self.text
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        return cls(
            segments=[Segment.from_dict(s) for s in d["segments"]],
            language=d.get("language"),
            duration=d["duration"],
            source=AudioInfo(**d["source"]) if d.get("source") else None,
            metadata=d.get("metadata", {}),
        )
