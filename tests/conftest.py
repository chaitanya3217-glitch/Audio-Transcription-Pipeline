import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

SR = 16_000


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def tone_with_gaps(seconds: float, gap_every: float = 7.0, gap_len: float = 1.0) -> np.ndarray:
    """A 440 Hz tone ("speech") interrupted by silent gaps ("pauses")."""
    t = np.arange(int(seconds * SR)) / SR
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig[(t % gap_every) > gap_every - gap_len] = 0.0
    return sig.astype(np.float32)


@pytest.fixture
def speechy_wav(tmp_path) -> Path:
    return write_wav(tmp_path / "speechy.wav", tone_with_gaps(95))


@pytest.fixture
def to_format(tmp_path):
    """Transcode a file into another container/codec with ffmpeg."""
    def convert(src: Path, ext: str, extra: list[str] | None = None) -> Path:
        dst = tmp_path / f"converted{ext}"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), *(extra or []),
                        str(dst)], check=True)
        return dst
    return convert
