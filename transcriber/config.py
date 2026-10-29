"""Runtime settings, read from environment variables (12-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"TRANSCRIBER_{name}", default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    model: str = field(default_factory=lambda: _env("MODEL", "base"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    compute_type: str = field(default_factory=lambda: _env("COMPUTE_TYPE", "int8"))
    workers: int = field(default_factory=lambda: int(_env("WORKERS", "2")))
    max_queue: int = field(default_factory=lambda: int(_env("MAX_QUEUE", "100")))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "500")))
    chunk_seconds: float = field(default_factory=lambda: float(_env("CHUNK_SECONDS", "600")))
    max_job_attempts: int = field(default_factory=lambda: int(_env("MAX_JOB_ATTEMPTS", "3")))
