"""Type stubs for the ``mosqito`` perceptual-metric subset used by this package.

``mosqito`` ships without type information.  This stub (discovered via
``mypy_path = "src:stubs"``) lets ``mypy`` type-check
:mod:`auditory_handshake.metrics.perceptual` against the real ``mosqito``
API.  Only the functions the package calls are declared, and with minimal
type detail (``Any``) because ``mosqito`` returns a mix of scalars and
time-varying arrays.
"""

from typing import Any

# loudness_zwtv(signal, fs, field_type='free')
# -> (N [overall loudness, per time], N_specific [bark x time], bark_axis, time_axis)
def loudness_zwtv(
    signal: Any, fs: float, field_type: str = "free"
) -> tuple[Any, Any, Any, Any]: ...

# sharpness_din_st(signal, fs) -> S in acum (scalar, or an array over time)
def sharpness_din_st(signal: Any, fs: float) -> Any: ...

# roughness_dw(signal, fs, overlap=0.5)
# -> (R [overall roughness, per segment], R_specific [bark x time], bark_axis, time)
def roughness_dw(signal: Any, fs: float, overlap: float = 0.5) -> tuple[Any, Any, Any, Any]: ...
