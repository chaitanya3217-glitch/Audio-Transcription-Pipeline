"""Error taxonomy.

The split matters for recovery: a *permanent* error (corrupt file, no audio
stream) will fail the same way every time, so retrying only wastes compute.
A *transient* error (OOM, model hiccup, disk blip) is worth retrying.
"""


class TranscriptionError(Exception):
    """Base class for all pipeline errors."""


class PermanentError(TranscriptionError):
    """Retrying will not help."""


class UnsupportedAudioError(PermanentError):
    """The input is not decodable audio (or has no audio stream)."""


class TransientError(TranscriptionError):
    """Worth retrying."""
