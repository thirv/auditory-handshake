"""Perceptual acoustic metrics (loudness, sharpness, roughness).

This module adds *perceptual* quality metrics on top of the physical ones in
:mod:`auditory_handshake.metrics`.  The metrics are computed with the
:mod:`mosqito` toolkit (https://pypi.org/project/mosqito/), which implements
the ISO 532 loudness and DIN sharpness/roughness models:

* **Loudness** (sones) -- the perceived volume (ISO 532).
* **Sharpness** (acum) -- the "sharp/bright" quality (DIN 45692).
* **Roughness** (asper) -- the perceptual "raspiness" of amplitude-modulated
  components (Daniel & Weber).

The :class:`~auditory_handshake.metrics.base.PerceptualMetrics` dataclass
lives in :mod:`auditory_handshake.metrics.base` so the whole metrics API is
importable from the package root without circularity; this module only
provides the *implementation*.

``mosqito`` is imported lazily inside the functions so that the rest of the
package does not pay for the dependency at import time.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from auditory_handshake.metrics.base import PerceptualAnnoyance, PerceptualMetrics

__all__ = ["perceptual_metrics", "perceptual_annoyance", "noise_source_perceptual_metrics"]

#: Minimum signal length (in samples) before the perceptual models are
#: evaluated.  ``mosqito``'s roughness model uses 200 ms sliding windows and
#: its temporal weighting needs enough frames to stabilise; below this the
#: steady-state metrics are undefined, so :func:`perceptual_metrics` reports
#: zeros for such (e.g. empty or sub-second) inputs instead of failing.
_MIN_SAMPLES = 48_000


def _zero_metrics() -> PerceptualMetrics:
    """An all-zero result, used for empty / short / degenerate signals."""
    return PerceptualMetrics(loudness_sone=0.0, sharpness_acum=0.0, roughness_asper=0.0)


def _steady_state(values: Any) -> float:
    """Reduce a time-varying metric array to a single steady-state value.

    ``mosqito`` reports loudness, sharpness and roughness as time-varying
    arrays (with transient edges from temporal weighting and the 200 ms
    roughness segments).  For a stationary signal the interior is flat, so the
    *median* over the finite frames is a robust steady-state estimate that is
    insensitive to the transient edges.  A signal with no finite frames (e.g.
    all-NaN) yields ``0.0``.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def _finite(values: Any) -> np.ndarray:
    """Return the finite entries of a time-varying array as a flat array."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _p95(values: Any) -> float:
    """95th percentile over finite frames (``0.0`` if none).  Peak annoyance."""
    a = _finite(values)
    return float(np.percentile(a, 95)) if a.size else 0.0


def _mean(values: Any) -> float:
    """Mean over finite frames (``0.0`` if none).  Sustained annoyance."""
    a = _finite(values)
    return float(np.mean(a)) if a.size else 0.0


# Reference perceptual levels for the annoyance *penalty*.  A sound at or below
# a reference is "tolerable" (penalty 0); above it the penalty ramps smoothly
# and monotonically to ~1.  The loudness reference is per noise type (broadband
# vs pink); sharpness and roughness each have a single reference.
_NOISE_N_REF = {"broadband": 30.0, "pink": 25.0}  # sone
_S_REF_ACUM = 8.0  # sharpness (acum)
_R_REF_ASPER = 2.0  # roughness (asper)
_SIGMOID_STEEPNESS = 0.5  # how quickly the penalty ramps past the reference
#: Signal types handled by the *reference-ramp* penalty (``noise`` / ``sharp`` /
#: ``rough``).
_RAMP_SIGNAL_TYPES = ("noise", "sharp", "rough")
#: Modulation signal types scored by the *frequency-dependent* annoyance
#: formula (see :func:`_modulation_cost`).
_MODULATION_SIGNAL_TYPES = ("css", "mfsk")

#: ``(alpha, beta)`` annoyance-coefficient pairs per modulation type.  CSS
#: (chirp-spaced) signals are perceptually dominated by sharpness (``alpha``),
#: MFSK by roughness (``beta``).
_MODULATION_COEFFS = {
    "css": (0.65, 0.35),
    "mfsk": (0.40, 0.60),
    #: Fallback when an unlisted modulation type is requested.
    "default": (0.50, 0.50),
}

#: Every ``signal_type`` accepted by :func:`perceptual_annoyance`.
_SIGNAL_TYPES = _RAMP_SIGNAL_TYPES + _MODULATION_SIGNAL_TYPES


def _validate_signal_type(signal_type: str, noise_kind: str) -> None:
    """Raise ``ValueError`` for an unrecognised ``signal_type``/``noise_kind``."""
    if signal_type not in _SIGNAL_TYPES:
        raise ValueError(f"signal_type must be one of {_SIGNAL_TYPES}, got {signal_type!r}")
    if signal_type == "noise" and noise_kind not in _NOISE_N_REF:
        raise ValueError(f"noise_kind must be one of {tuple(_NOISE_N_REF)}, got {noise_kind!r}")


def _penalty_ramp(value: float, ref: float) -> float:
    """Smooth monotone ramp: 0 at ``value <= ref``, rising to ~1 above it.

    A sigmoid centred at the reference, clamped to ``[0, 1]``, so the penalty
    is exactly ``0`` at and below the reference (a "tolerable" sound) and
    grows smoothly to its full value (~1) above it.  Monotone non-decreasing
    in ``value``.
    """
    z = _SIGMOID_STEEPNESS * (value - ref)
    # expit(z) = 1 / (1 + exp(-z)), evaluated overflow-safe for large |z|.
    if z >= 0.0:
        sig = 1.0 / (1.0 + math.exp(-z))
    else:
        sig = math.exp(z) / (1.0 + math.exp(z))
    penalty = 2.0 * (sig - 0.5)
    # The odd sigmoid alone would go negative below the reference, which would
    # make ``annoyance`` negative; clamp so "at/below reference" means 0.
    return float(min(1.0, max(0.0, penalty)))


def _modulation_factor(signal_type: str, S_avg: float, R_avg: float) -> float:
    """Frequency-dependent annoyance factor for a *modulated* signal.

    CSS and MFSK are discrete-carrier modulations, not broadband noise, so the
    reference-ramp penalty is the wrong model.  Instead the annoyance is the
    loudness scale multiplied by a factor that grows quadratically with the
    sharpness (``S_avg``) and roughness (``R_avg``) content::

        factor = 1.0 + (alpha * S_avg)^2 + (beta * R_avg)^2

    The ``alpha``/``beta`` weighting is per modulation type (see
    ``_MODULATION_COEFFS``): CSS (chirp-spaced) is dominated by sharpness
    (``alpha = 0.65``), MFSK by roughness (``beta = 0.60``); an unlisted
    modulation type falls back to ``alpha = beta = 0.50``.  Unlike the ramp
    penalty this factor is ``>= 1`` (it can exceed ``N_avg``).
    """
    alpha, beta = _MODULATION_COEFFS.get(signal_type, _MODULATION_COEFFS["default"])
    return 1.0 + (alpha * S_avg) ** 2 + (beta * R_avg) ** 2


def _annoyance_penalty(
    signal_type: str,
    noise_kind: str,
    N_avg: float,
    S_avg: float,
    R_avg: float,
) -> float:
    """Penalty multiplier for a signal type.

    The driving perceptual metric is chosen by ``signal_type`` (loudness for
    noise, sharpness for sharp, roughness for rough); for noise the reference
    is chosen by ``noise_kind``.  The overall annoyance is ``N_avg * penalty``
    (see :func:`perceptual_annoyance`).

    For the reference-ramp types (``noise`` / ``sharp`` / ``rough``) the
    penalty is ``0`` at/below the reference and ramps smoothly to ~1 above it.
    For the modulation types (``css`` / ``mfsk``) the penalty is instead the
    frequency-dependent factor :func:`_modulation_factor` (``>= 1``).
    """
    if signal_type == "noise":
        return _penalty_ramp(N_avg, _NOISE_N_REF[noise_kind])
    if signal_type == "sharp":
        return _penalty_ramp(S_avg, _S_REF_ACUM)
    if signal_type == "rough":
        return _penalty_ramp(R_avg, _R_REF_ASPER)
    if signal_type in _MODULATION_SIGNAL_TYPES:
        return _modulation_factor(signal_type, S_avg, R_avg)
    # Unreachable: callers validate with _validate_signal_type first.
    raise ValueError(f"signal_type must be one of {_SIGNAL_TYPES}, got {signal_type!r}")


def _validate_fs(fs: float) -> float:
    """Coerce ``fs`` to ``float`` and require a positive finite sampling rate."""
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be a positive finite sampling rate (Hz), got {fs!r}")
    return fs


def perceptual_metrics(x: npt.ArrayLike, fs: float) -> PerceptualMetrics:
    """Compute perceptual loudness, sharpness and roughness of a signal.

    Parameters
    ----------
    x:
        A 1-D pressure waveform (a tone, a handshake waveform, or a synthesized
        noise source).
    fs:
        The sampling frequency in Hz.

    Returns
    -------
    PerceptualMetrics
        ``loudness_sone`` (ISO 532), ``sharpness_acum`` (DIN 45692) and
        ``roughness_asper`` (Daniel & Weber).  Empty, very short (fewer than
        ~3 s of audio), or all-NaN signals return an all-zero result rather
        than raising.

    Raises
    ------
    ValueError
        If ``fs`` is not a positive finite sampling rate.
    """
    fs = _validate_fs(fs)
    x = np.asarray(x, dtype=float).ravel()
    if x.size < _MIN_SAMPLES:
        return _zero_metrics()

    # Lazy import: keep the heavy dependency out of the package import path.
    from mosqito.sq_metrics import loudness_zwtv, roughness_dw, sharpness_din_st

    try:
        # loudness_zwtv returns (N, N_spec, bark_axis, time_axis); the leading
        # element N is the sones-vs-time array we want.
        loudness = _steady_state(loudness_zwtv(x, fs)[0])
        # sharpness_din_st returns a scalar S in acum.
        sharpness = _steady_state(sharpness_din_st(x, fs))
        # roughness_dw returns (R, R_spec, bark, time); R is asper-vs-time.
        roughness = _steady_state(roughness_dw(x, fs)[0])
    except Exception:  # degenerate inputs (e.g. all-NaN) -> all-zero result
        return _zero_metrics()

    # Guard against any non-finite value that slipped through (e.g. a fully
    # silent input can still produce NaN in the internal spectra); fall back to
    # the all-zero result so callers always get a finite, well-defined value.
    if not all(np.isfinite(v) for v in (loudness, sharpness, roughness)):
        return _zero_metrics()

    return PerceptualMetrics(
        loudness_sone=float(loudness),
        sharpness_acum=float(sharpness),
        roughness_asper=float(roughness),
    )


def _zero_annoyance() -> PerceptualAnnoyance:
    """An all-zero annoyance result, used for empty / short / degenerate signals."""
    return PerceptualAnnoyance(N_5=0.0, N_avg=0.0, S_avg=0.0, R_avg=0.0, annoyance=0.0)


def perceptual_annoyance(
    x: npt.ArrayLike,
    fs: float,
    signal_type: str = "noise",
    noise_kind: str = "broadband",
) -> PerceptualAnnoyance:
    """Reduce a signal to a single scalar perceptual-annoyance cost.

    The cost is a *multiplicative* model: the sustained loudness ``N_avg``
    (sone) is multiplied by a dimensionless *penalty* that captures how
    annoying the sound is beyond a "tolerable" reference level.  The penalty
    is ``0`` at and below the reference and ramps smoothly and monotonically
    to ~1 above it (a sigmoid clamped to ``[0, 1]``).  A quiet / tolerable
    sound therefore scores exactly ``0``, and a loud, sharp or rough one
    scores up to ``N_avg`` (hence ``0 ≤ annoyance ≤ N_avg``).

    The penalty is driven by ``signal_type``:

    - ``"noise"`` (e.g. broadband or pink noise): the penalty ramps with
      loudness; the reference is chosen by ``noise_kind`` (``"broadband"`` =
      30 sone, ``"pink"`` = 25 sone).
    - ``"sharp"``: the penalty ramps with sharpness (reference 8 acum).
    - ``"rough"``: the penalty ramps with roughness (reference 2 asper).
    - ``"css"`` / ``"mfsk"`` (modulated carriers): the penalty is a
      frequency-dependent factor ``1 + (alpha*S_avg)^2 + (beta*R_avg)^2``
      (>= 1); CSS weights sharpness (alpha=0.65, beta=0.35), MFSK weights
      roughness (alpha=0.40, beta=0.60).

    Parameters
    ----------
    x:
        A 1-D pressure waveform (a tone, a handshake waveform, or a synthesized
        noise source).
    fs:
        The sampling frequency in Hz.
    signal_type:
        One of ``"noise"``, ``"sharp"``, ``"rough"``, ``"css"``, ``"mfsk"`` --
        selects which perceptual metric (loudness / sharpness / roughness)
        drives the penalty, or the frequency-dependent modulation factor.
    noise_kind:
        ``"broadband"`` or ``"pink"``; selects the loudness reference when
        ``signal_type`` is ``"noise"`` (ignored otherwise).

    Returns
    -------
    PerceptualAnnoyance
        Peak (``N_5``) and mean (``N_avg``) loudness (sone), mean sharpness
        (``S_avg``, acum) and roughness (``R_avg``, asper) -- all from
        :mod:`mosqito` -- and ``annoyance`` = ``N_avg * penalty``.  Empty, very
        short (fewer than ~3 s of audio), or all-NaN signals return an
        all-zero result rather than raising, matching :func:`perceptual_metrics`.

    Raises
    ------
    ValueError
        If ``fs`` is not a positive finite sampling rate, or ``signal_type`` /
        ``noise_kind`` is not a recognised value.
    """
    fs = _validate_fs(fs)
    _validate_signal_type(signal_type, noise_kind)
    x = np.asarray(x, dtype=float).ravel()
    if x.size < _MIN_SAMPLES:
        return _zero_annoyance()

    # Lazy import: keep the heavy dependency out of the package import path.
    from mosqito.sq_metrics import loudness_zwtv, roughness_dw, sharpness_din_st

    try:
        # Time-varying loudness (sone), sharpness (acum) and roughness (asper).
        loudness = loudness_zwtv(x, fs)[0]
        sharpness = sharpness_din_st(x, fs)
        roughness = roughness_dw(x, fs)[0]

        # Peak (95th percentile) and mean loudness; mean sharpness and roughness.
        N_5 = _p95(loudness)
        N_avg = _mean(loudness)
        S_avg = _mean(sharpness)
        R_avg = _mean(roughness)
    except Exception:  # degenerate inputs (e.g. all-NaN) -> zero annoyance
        return _zero_annoyance()

    if not all(np.isfinite(v) for v in (N_5, N_avg, S_avg, R_avg)):
        return _zero_annoyance()

    N_avg = float(N_avg)
    S_avg = float(S_avg)
    R_avg = float(R_avg)
    penalty = _annoyance_penalty(signal_type, noise_kind, N_avg, S_avg, R_avg)
    return PerceptualAnnoyance(
        N_5=float(N_5),
        N_avg=N_avg,
        S_avg=S_avg,
        R_avg=R_avg,
        annoyance=N_avg * penalty,
    )


def noise_source_perceptual_metrics(
    noise_source: Any,
    duration_sec: float,
    fs: float,
    *,
    seed: int = 42,
    salt: int = 0,
) -> PerceptualMetrics:
    """Perceptual metrics for a :class:`~auditory_handshake.config.NoiseSource`.

    Synthesizes the source waveform for ``duration_sec`` seconds at ``fs`` via
    :func:`auditory_handshake.acoustics.noise.synthesize_noise` and returns
    :func:`perceptual_metrics` on the result -- i.e. it is exactly equivalent
    to::

        perceptual_metrics(
            synthesize_noise(noise_source, duration_sec, fs, seed=seed, salt=salt),
            fs,
        )

    Parameters
    ----------
    noise_source:
        A :class:`auditory_handshake.config.NoiseSource` describing the sound
        (kind, level, distance, ...).
    duration_sec:
        How long to synthesize the source in seconds.
    fs:
        The sampling frequency in Hz.
    seed, salt:
        Seed / salt forwarded to :func:`synthesize_noise` for reproducibility.
    """
    from auditory_handshake.acoustics.noise import synthesize_noise

    x = synthesize_noise(noise_source, duration_sec, fs, seed=seed, salt=salt)
    return perceptual_metrics(x, fs)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    import numpy as np

    from auditory_handshake.config import NoiseSource

    fs = 48000
    t = np.arange(fs) / fs  # 1 s
    tone = 0.00002 * 10 ** (60 / 20) * np.sin(2 * np.pi * 1000 * t)  # 1 kHz, ~60 dB SPL
    pm = perceptual_metrics(tone, fs)
    print(
        "1 kHz tone : loudness="
        f"{pm.loudness_sone:.2f} sones, sharpness={pm.sharpness_acum:.3f} acum, "
        f"roughness={pm.roughness_asper:.4f} asper"
    )
    src = NoiseSource(kind="broadband", level_db_spl=70.0, distance_m=1.0)
    nsm = noise_source_perceptual_metrics(src, 1.0, fs)
    print(
        "broadband  : loudness="
        f"{nsm.loudness_sone:.2f} sones, sharpness={nsm.sharpness_acum:.3f} acum, "
        f"roughness={nsm.roughness_asper:.4f} asper"
    )
    sys.exit(0)
