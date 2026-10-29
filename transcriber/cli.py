"""Command line entry point.

    python -m transcriber transcribe samples/short.mp3 --format srt
    python -m transcriber serve --port 8000
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from .engine import WhisperEngine
from .errors import TranscriptionError
from .pipeline import PipelineConfig, TranscriptionPipeline
from .postprocess import FORMATTERS


def _transcribe(args: argparse.Namespace) -> int:
    engine = WhisperEngine(args.model, args.device, args.compute_type,
                           word_timestamps=args.word_timestamps)
    pipeline = TranscriptionPipeline(engine, PipelineConfig(
        language=args.language, chunk_seconds=args.chunk_seconds))

    def progress(done: int, total: int) -> None:
        print(f"  chunk {done}/{total} done", file=sys.stderr)

    try:
        if args.work_dir:  # persistent work dir -> re-running resumes from checkpoints
            result = pipeline.run(args.input, args.work_dir, progress)
        else:
            with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
                result = pipeline.run(args.input, tmp, progress)
    except TranscriptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = FORMATTERS[args.format](result)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(output)
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .api import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcriber")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe one audio file")
    t.add_argument("input")
    t.add_argument("-f", "--format", choices=sorted(FORMATTERS), default="json")
    t.add_argument("-o", "--output", help="write to file instead of stdout")
    t.add_argument("-l", "--language", help="e.g. 'en'; default auto-detect")
    t.add_argument("-m", "--model", default="base",
                   help="tiny|base|small|medium|large-v3 (accuracy vs speed)")
    t.add_argument("--device", default="auto")
    t.add_argument("--compute-type", default="int8")
    t.add_argument("--chunk-seconds", type=float, default=600.0)
    t.add_argument("--word-timestamps", action="store_true")
    t.add_argument("--work-dir", help="keep intermediates here (enables resume)")
    t.set_defaults(func=_transcribe)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
