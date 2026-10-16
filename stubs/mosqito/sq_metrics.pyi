"""Type stubs for ``mosqito.sq_metrics`` (installed 1.2.1, not type-annotated).

Only the functions imported by ``auditory_handshake`` are declared here, so
the package type-checks under mypy ``strict`` without needing mosqito itself
to ship ``py.typed`` metadata.
"""
from typing import Any

__all__ = ["loudness_zwtv", "roughness_dw", "sharpness_din_st"]
def loudness_zwtv(x: Any, fs: float, **kwargs: Any) -> Any: ...
def roughness_dw(x: Any, fs: float, **kwargs: Any) -> Any: ...
def sharpness_din_st(x: Any, fs: float, **kwargs: Any) -> Any: ...
