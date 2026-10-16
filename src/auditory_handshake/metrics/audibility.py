"""Perceptual audibility (C2): A-weighting x Fletcher-Munson threshold.

Band-dependent, SPL-monotone, non-degenerate perceptual cost (PLAN.md "audibility
score") so the optimiser can't trade audibility for capacity. The score is the
A-weighted tone level above the 0-phon hearing threshold at its frequency:

    audibility_cost_db(f, L) = (L + A(f)) - threshold(f)   [dB]

- ``A(f)`` — :func:`a_weighting_db` — IEC 6167 A-weighting (dB), 0 dB at 1 kHz
  (librosa when importable, else an identical native implementation).
- ``threshold(f)`` — :func:`hearing_threshold_db` — Fletcher-Munson 0-phon
  hearing threshold (dB SPL), log-linear interpolation of the standard table.

Run ``python -m auditory_handshake.metrics.audibility`` for a band table (C2
checkpoint: cost is band-dependent and SPL-monotone, i.e. non-degenerate).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["a_weighting_db", "hearing_threshold_db", "audibility_cost_db"]


# --------------------------------------------------------------------------- #
# IEC 6167 A-weighting (pluggable backend: librosa preferred, native fallback)
# --------------------------------------------------------------------------- #
#: IEC 6167 A-weighting corner frequencies (Hz): f1, f2, f3, f4.
_IEC6167_CORNERS_HZ = (12194.217, 20.598997, 107.65265, 737.86223)


def _a_weighting_native(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    """Native IEC 6167 A-weighting (dB); numerically identical to librosa."""
    f_sq = frequencies**2.0
    const = np.asarray(_IEC6167_CORNERS_HZ, dtype=np.float64) ** 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 2.0 + 20.0 * (
            np.log10(const[0])
            + 2.0 * np.log10(f_sq)
            - np.log10(f_sq + const[0])
            - np.log10(f_sq + const[1])
            - 0.5 * np.log10(f_sq + const[2])
            - 0.5 * np.log10(f_sq + const[3])
        )
    return np.asarray(weights, dtype=np.float64)


_A_WEIGHTING_BACKEND: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None


def _resolve_a_weighting_backend() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Pick the A-weighting backend once: librosa if importable, else native."""
    global _A_WEIGHTING_BACKEND
    if _A_WEIGHTING_BACKEND is None:
        try:
            librosa = importlib.import_module("librosa")
        except ModuleNotFoundError:
            _A_WEIGHTING_BACKEND = _a_weighting_native
        else:

            def _librosa_backend(freqs: NDArray[np.float64]) -> NDArray[np.float64]:
                return np.asarray(librosa.A_weighting(freqs, min_db=None), dtype=np.float64)

            _A_WEIGHTING_BACKEND = _librosa_backend
    return _A_WEIGHTING_BACKEND


def a_weighting_db(
    frequencies: float | Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """IEC 6167 A-weighting (dB), 0 dB at 1 kHz.

    Uses ``librosa.A_weighting`` when available, else an identical native
    IEC 6167 implementation, so the dependency stays optional and the result
    is the same either way.
    """
    return _resolve_a_weighting_backend()(np.asanyarray(frequencies, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Fletcher-Munson 0-phon hearing threshold
# --------------------------------------------------------------------------- #
#: Standard 0-phon (Fletcher-Munson) hearing threshold in dB SPL. 1 kHz = 0 dB
#: SPL by definition of the phon; near-flat across ~125 Hz-8 kHz; rising
#: sharply above ~8 kHz (near-ultrasonic penalty) and moderately below ~100 Hz.
_FREQUENCY_HZ = (
    20.0,
    30.0,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
    25000.0,
)
_THRESHOLD_DB_SPL = (
    20.0,
    14.0,
    10.0,
    7.0,
    4.0,
    2.5,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.0,
    15.0,
    30.0,
    45.0,
    85.0,
)
#: Pre-computed log10(f) nodes for log-linear interpolation.
_THRESHOLD_LOG10_F = np.log10(np.asarray(_FREQUENCY_HZ, dtype=np.float64))
_THRESHOLD_TABLE = np.asarray(_THRESHOLD_DB_SPL, dtype=np.float64)


def hearing_threshold_db(
    frequencies: float | Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fletcher-Munson 0-phon hearing threshold (dB SPL) at ``frequencies``.

    Log-linear (linear-in-log10-f) interpolation of the standard 0-phon table.
    Frequencies below the 20 Hz node hold the 20 Hz value; above 25 kHz hold
    the 25 kHz value.
    """
    f = np.asanyarray(frequencies, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError(
            f"frequencies must be > 0 Hz (got min={float(f.min()) if f.size else 'empty'})"
        )
    return np.interp(np.log10(f), _THRESHOLD_LOG10_F, _THRESHOLD_TABLE)


# --------------------------------------------------------------------------- #
# Combined audibility score
# --------------------------------------------------------------------------- #
def audibility_cost_db(frequency_hz: float, level_db: float) -> float:
    """Perceptual audibility/annoyance cost (dB) of a pure tone.

    The A-weighted level ``L + A(f)`` above the 0-phon hearing threshold
    ``threshold(f)`` at frequency ``frequency_hz``. Higher = more audible =
    more penalised in the maximise-utility objective. The cost is

    - **band-dependent**: A(f) and threshold(f) vary with frequency;
    - **monotone in SPL**: strictly increasing in ``level_db``;
    - **non-degenerate**: distinct bands have distinct costs at any fixed SPL.
    """
    f = float(frequency_hz)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError(f"frequency_hz must be > 0 (got {frequency_hz})")
    lvl = float(level_db)
    if not np.isfinite(lvl):
        raise ValueError(f"level_db must be finite (got {level_db})")
    a = float(a_weighting_db([f])[0])
    t = float(hearing_threshold_db([f])[0])
    return float(lvl - t + a)


# --------------------------------------------------------------------------- #
# Demo (``python -m auditory_handshake.metrics.audibility``)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    print("C2 — audibility score: audibility_cost_db(f, L) = L + A(f) - threshold(f)")
    print("=" * 68)
    levels = (70, 80, 90)
    bands = (100, 1000, 2000, 4000, 8000, 16000, 20000)
    print("band (Hz)   " + "".join(f"{lvl:>8} dB  " for lvl in levels))
    for band in bands:
        row = "".join(f"{audibility_cost_db(band, lvl):>10.2f} " for lvl in levels)
        print(f"{band:>10}   {row}")
    print()
    print("C2 checks:")
    print(
        f"  band-dependent (non-degenerate) at 80 dB: "
        f"{len({round(audibility_cost_db(b, 80), 1) for b in bands})} distinct values"
    )
    print(
        f"  monotone in SPL (1 kHz): "
        f"{[round(audibility_cost_db(1000, lvl), 1) for lvl in levels]}"
    )


if __name__ == "__main__":
    _demo()
