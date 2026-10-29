"""Audio I/O: probing, normalization and silence-aware chunking.

Design: every input, whatever its container/codec, is converted ONCE by ffmpeg
into a canonical 16 kHz mono 16-bit PCM WAV. Everything downstream (chunking,
the ASR engine) only ever sees that one format. This keeps format handling in
a single place and makes the WAV randomly seekable, so we can read any time
window without decoding the whole file into memory.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .errors import TransientError, UnsupportedAudioError
from .models import AudioInfo, Chunk

SAMPLE_RATE = 16_000  # what Whisper-family models expect

# Informational only: we trust ffprobe's view of the *content*, not the file
# extension. Anything ffmpeg can decode (incl. audio tracks of video files) works.
COMMON_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
                     ".webm", ".wma", ".amr", ".aiff", ".mp4", ".mkv", ".mov"}


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f"'{binary}' not found on PATH; install ffmpeg")
    return path


def probe(path: str | Path) -> AudioInfo:
    """Inspect a file with ffprobe. Raises UnsupportedAudioError if it holds no audio."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise UnsupportedAudioError(f"file is missing or empty: {path.name}")

    cmd = [_require("ffprobe"), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", "-select_streams", "a", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.replace(str(path), path.name).strip()  # never leak server paths
        raise UnsupportedAudioError(f"not a decodable media file: {err[:300]}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise UnsupportedAudioError("file contains no audio stream")

    stream, fmt = streams[0], data.get("format", {})
    return AudioInfo(
        format_name=fmt.get("format_name", "unknown"),
        codec=stream.get("codec_name", "unknown"),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        duration=float(fmt.get("duration") or stream.get("duration") or 0.0),
        size_bytes=path.stat().st_size,
    )


def normalize(src: str | Path, dst: str | Path) -> float:
    """Decode `src` to 16 kHz mono PCM WAV at `dst`. Returns the exact duration in seconds.

    Writes to a temp name and renames, so a crash never leaves a half-written
    file that a resumed job would mistake for a finished one.
    """
    dst = Path(dst)
    tmp = dst.with_suffix(".partial.wav")
    cmd = [_require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-c:a", "pcm_s16le", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        err = proc.stderr.replace(str(src), Path(src).name).strip()
        raise UnsupportedAudioError(f"ffmpeg failed to decode audio: {err[:300]}")
    tmp.replace(dst)
    return wav_duration(dst)


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def read_window(path: str | Path, start: float, end: float) -> np.ndarray:
    """Read [start, end) seconds of a normalized WAV as float32 in [-1, 1].

    Only that window is loaded, so memory stays bounded for arbitrarily long files.
    """
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            first = max(0, int(start * sr))
            count = max(0, min(w.getnframes(), int(end * sr)) - first)
            w.setpos(first)
            raw = w.readframes(count)
    except (OSError, EOFError, wave.Error) as exc:
        raise TransientError(f"could not read audio window: {exc}") from exc
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- #
# Long-audio chunking
# --------------------------------------------------------------------------- #

def quietest_offset(samples: np.ndarray, sr: int = SAMPLE_RATE,
                    frame_s: float = 0.02, smooth_s: float = 0.4) -> float:
    """Offset (seconds) of the centre of the quietest `smooth_s`-long stretch of `samples`.

    Pure function: frame RMS energy -> moving average -> argmin.
    """
    frame = max(1, int(frame_s * sr))
    n_frames = len(samples) // frame
    if n_frames == 0:
        return len(samples) / sr / 2
    energy = np.sqrt(np.mean(samples[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    k = max(1, min(n_frames, int(smooth_s / frame_s)))
    smoothed = np.convolve(energy, np.ones(k) / k, mode="valid")
    return (int(np.argmin(smoothed)) + k / 2) * frame / sr


def plan_chunks(wav_path: str | Path, duration: float, chunk_s: float,
                search_s: float = 30.0) -> list[Chunk]:
    """Split audio into ~chunk_s pieces, cutting at the quietest point near each boundary.

    For every target boundary we look back over the preceding `search_s`
    seconds and cut in the lowest-energy stretch (usually a pause between
    sentences), so we almost never split a word. Because cuts land in silence
    the chunks do not need to overlap, which means no de-duplication step.
    """
    if duration <= chunk_s:
        return [Chunk(0, 0.0, duration)]

    bounds, prev = [0.0], 0.0
    while duration - prev > chunk_s:
        target = prev + chunk_s
        win_start = max(prev + chunk_s / 2, target - search_s)  # never make tiny chunks
        window = read_window(wav_path, win_start, target)
        cut = win_start + quietest_offset(window)
        bounds.append(cut)
        prev = cut
    bounds.append(duration)
    return [Chunk(i, s, e) for i, (s, e) in enumerate(zip(bounds, bounds[1:]))]
