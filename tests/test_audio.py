import numpy as np
import pytest

from transcriber import audio
from transcriber.errors import UnsupportedAudioError

from .conftest import SR, tone_with_gaps, write_wav


@pytest.mark.parametrize("ext,extra", [
    (".mp3", []),
    (".flac", []),
    (".ogg", ["-c:a", "libvorbis"]),
    (".m4a", ["-c:a", "aac"]),
    (".wav", ["-ar", "44100", "-ac", "2"]),  # stereo 44.1 kHz -> must become mono 16 kHz
])
def test_normalize_any_format_to_16k_mono(speechy_wav, to_format, tmp_path, ext, extra):
    src = to_format(speechy_wav, ext, extra)
    info = audio.probe(src)
    assert info.duration == pytest.approx(95, abs=0.2)

    out = tmp_path / "norm.wav"
    duration = audio.normalize(src, out)
    assert duration == pytest.approx(95, abs=0.2)
    import wave
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)


def test_extension_is_not_trusted(speechy_wav, to_format, tmp_path):
    mp3 = to_format(speechy_wav, ".mp3")
    disguised = tmp_path / "actually_mp3.wav"
    disguised.write_bytes(mp3.read_bytes())
    assert audio.probe(disguised).codec == "mp3"


@pytest.mark.parametrize("content", [b"", b"this is not audio at all" * 100])
def test_rejects_non_audio(tmp_path, content):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(content)
    with pytest.raises(UnsupportedAudioError):
        audio.probe(bad)


def test_read_window_returns_only_requested_slice(speechy_wav):
    samples = audio.read_window(speechy_wav, 10.0, 12.5)
    assert len(samples) == int(2.5 * SR)
    assert samples.dtype == np.float32 and np.abs(samples).max() <= 1.0


def test_quietest_offset_finds_the_gap():
    sig = 0.5 * np.ones(SR * 10, dtype=np.float32)
    sig[int(6.0 * SR): int(6.6 * SR)] = 0.0   # silence centred at 6.3 s
    assert audio.quietest_offset(sig) == pytest.approx(6.3, abs=0.1)


def test_short_audio_is_one_chunk(speechy_wav):
    chunks = audio.plan_chunks(speechy_wav, 95.0, chunk_s=600)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 95.0)]


def test_long_audio_cuts_land_in_silence_and_cover_everything(speechy_wav):
    chunks = audio.plan_chunks(speechy_wav, 95.0, chunk_s=30, search_s=10)
    assert len(chunks) >= 3
    assert chunks[0].start == 0.0 and chunks[-1].end == 95.0
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == b.start                      # contiguous, no gaps/overlap
        assert a.duration <= 30 + 1e-6
        # gaps are at t % 7 in (6, 7): every cut should fall inside one
        assert 6.0 <= a.end % 7.0 <= 7.0, a.end
