"""Turn raw ASR output into something clean for downstream consumers."""
from __future__ import annotations

import json
import re

from .models import Segment, Transcript

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")

# Whisper's own heuristic for "this segment is probably a hallucination on silence".
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0


def clean_text(text: str) -> str:
    text = _WS.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def is_likely_hallucination(seg: Segment) -> bool:
    return (seg.no_speech_prob is not None and seg.avg_logprob is not None
            and seg.no_speech_prob > NO_SPEECH_THRESHOLD
            and seg.avg_logprob < LOGPROB_THRESHOLD)


def postprocess(segments: list[Segment]) -> list[Segment]:
    """Clean text, drop empty/hallucinated/repeated segments, enforce ordering, renumber."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        seg.text = clean_text(seg.text)
        if not seg.text or is_likely_hallucination(seg):
            continue
        if out and seg.text == out[-1].text:  # whisper repetition loop
            continue
        if out and seg.start < out[-1].end:   # clamp tiny overlaps from engine drift
            seg.start = out[-1].end
        seg.end = max(seg.end, seg.start)
        out.append(seg)
    for i, seg in enumerate(out):
        seg.id = i
        seg.start, seg.end = round(seg.start, 3), round(seg.end, 3)
    return out


# --------------------------------------------------------------------------- #
# Output formats
# --------------------------------------------------------------------------- #

def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(t: Transcript) -> str:
    blocks = [f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{s.text}\n"
              for i, s in enumerate(t.segments, 1)]
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    cues = [f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}\n{s.text}\n" for s in t.segments]
    return "WEBVTT\n\n" + "\n".join(cues)


def to_txt(t: Transcript) -> str:
    return "\n".join(f"[{_ts(s.start, '.')} - {_ts(s.end, '.')}] {s.text}" for s in t.segments) + "\n"


def to_json(t: Transcript) -> str:
    return json.dumps(t.to_dict(), indent=2, ensure_ascii=False)


FORMATTERS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_txt}
MEDIA_TYPES = {"json": "application/json", "srt": "application/x-subrip",
               "vtt": "text/vtt", "txt": "text/plain"}
