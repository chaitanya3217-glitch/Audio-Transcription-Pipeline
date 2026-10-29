import pytest

from transcriber.engine import MockEngine
from transcriber.errors import TransientError, UnsupportedAudioError
from transcriber.models import Segment, Transcript
from transcriber.pipeline import PipelineConfig, TranscriptionPipeline
from transcriber.postprocess import postprocess, to_srt, to_vtt


def make(engine=None, **cfg):
    return TranscriptionPipeline(engine or MockEngine(),
                                 PipelineConfig(retry_base_delay=0, **cfg))


def test_segments_are_on_the_original_timeline(speechy_wav, tmp_path):
    result = make(chunk_seconds=30).run(speechy_wav, tmp_path / "w")
    assert result.metadata["chunks"] >= 3
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    assert result.segments[-1].end <= result.duration
    assert max(s.end for s in result.segments) > 60  # later chunks were shifted
    assert [s.id for s in result.segments] == list(range(len(result.segments)))


def test_transient_failures_are_retried(speechy_wav, tmp_path):
    engine = MockEngine(fail_times=2)
    result = make(engine, max_attempts=3).run(speechy_wav, tmp_path / "w")
    assert result.segments and engine.calls == 3


def test_gives_up_after_max_attempts(speechy_wav, tmp_path):
    with pytest.raises(TransientError):
        make(MockEngine(fail_times=5), max_attempts=2).run(speechy_wav, tmp_path / "w")


def test_permanent_errors_are_not_retried(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"garbage" * 50)
    engine = MockEngine()
    with pytest.raises(UnsupportedAudioError):
        make(engine).run(bad, tmp_path / "w")
    assert engine.calls == 0


def test_resume_reuses_finished_chunks(speechy_wav, tmp_path):
    work = tmp_path / "w"
    first = make(chunk_seconds=30).run(speechy_wav, work)
    (work / "chunks" / "0001.json").unlink()        # simulate a crash mid-job

    engine = MockEngine()
    second = make(engine, chunk_seconds=30).run(speechy_wav, work)
    assert engine.calls == 1                        # only the missing chunk re-ran
    assert second.metadata["chunks_resumed_from_checkpoint"] == first.metadata["chunks"] - 1
    assert [s.text for s in second.segments] == [s.text for s in first.segments]


def test_postprocess_cleans_and_filters():
    segs = [
        Segment(0, 0.0, 2.0, "  hello   world  ."),
        Segment(1, 2.0, 4.0, "hello world."),                       # repetition loop
        Segment(2, 4.0, 6.0, "Thanks for watching!", -1.5, 0.9),    # silence hallucination
        Segment(3, 5.5, 8.0, "real speech"),
        Segment(4, 8.0, 9.0, "   "),
    ]
    out = postprocess(segs)
    assert [s.text for s in out] == ["hello world.", "real speech"]
    assert [s.id for s in out] == [0, 1]


def test_subtitle_formats():
    t = Transcript([Segment(0, 3661.5, 3662.25, "hi")], "en", 3700)
    assert "01:01:01,500 --> 01:01:02,250" in to_srt(t)
    assert to_vtt(t).startswith("WEBVTT") and "01:01:01.500" in to_vtt(t)
