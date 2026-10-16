"""Minimal type stubs for pyroomacoustics (no official stubs; see mypy config)."""

from collections.abc import Sequence
from typing import Any

class Material:
    def __init__(self, absorption_coeff: float | dict[str, Any] = ...) -> None: ...

class SoundSource:
    def __init__(
        self,
        position: Sequence[float] | None = ...,
        **kwargs: Any,
    ) -> None: ...

class ShoeBox:
    def __init__(
        self,
        dimensions: Sequence[float],
        fs: float = ...,
        materials: dict[str, Material] | None = ...,
        max_order: int = ...,
        **kwargs: Any,
    ) -> None: ...

    # Computed RIR: (num_mics, num_sources) of arrays; each entry indexed
    # [mic][source] is a 1-D impulse-response array.
    rir: Any

    def add_soundsource(self, *args: Any, **kwargs: Any) -> None: ...
    def add_microphone(self, *args: Any, **kwargs: Any) -> None: ...
    def compute_rir(self) -> None: ...
    def rt60_theory(self, formula: str = "sabine") -> float: ...
