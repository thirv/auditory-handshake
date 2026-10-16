"""Minimal type stubs for soundfile (no official stubs; see mypy config)."""

from typing import Any

def write(file: Any, data: Any, samplerate: int, **kwargs: Any) -> None: ...
