"""Multi-finger RAKE receiver: dominant-path extraction + maximum-ratio combining.

The acoustic channel is a (multipath) FIR filter whose impulse response is the
room RIR.  A RAKE receiver estimates the dominant paths and then re-combines
them coherently so that their energy collapses onto the dominant path,
suppressing the inter-path interference that otherwise hurts the downstream
CSS/MFSK demodulator (the "multipath compression" the sims A/B test).

This module is deliberately framework-free (numpy only) so the ``sim`` scripts
drive it directly:

* :class:`PathTap`  -- a single estimated path (delay, gain, power, RTT).
* :func:`find_paths` -- peak-pick the dominant taps of a normalised RIR.
* :func:`rake`       -- build a maximum-ratio FIR combiner from those taps and
  apply it to a received waveform.

Physics (PLAN.md §3.6 / §2): a path at delay ``d`` samples carries a *two-way*
round-trip time ``rtt = 2 * d / fs`` (the "TTCH = 2 * d / c" relation), and the
channel's TTCH is the time needed to coherently accumulate those paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Sampling rate used by :func:`find_paths` when the caller omits one.  Matches
# the CSS/MFSK default (``DEFAULT_FS = 48 kHz``) so ``tap.rtt_s`` /
# ``tap.delay_seconds`` are correct for the RIRs the sim scripts generate.
DEFAULT_FS = 48_000.0


@dataclass(frozen=True)
class PathTap:
    """A single estimated multipath (one RAKE finger).

    Attributes:
        delay_samples: Delay of this path in samples (index into the RIR).
        gain: Complex amplitude of the path, normalised so the strongest path
            has ``abs(gain) == 1``.  Real for the RIR-based sims here.
        fs: Sampling rate (Hz) used for the delay <-> time conversion.
    """

    delay_samples: int
    gain: complex
    fs: float

    @property
    def delay_seconds(self) -> float:
        """One-way delay of this path (seconds)."""
        return self.delay_samples / self.fs

    @property
    def power(self) -> float:
        """Path power (``|gain|^2``), normalised to the dominant path."""
        return float(abs(self.gain) ** 2)

    @property
    def rtt_s(self) -> float:
        """Round-trip time (RTT) of this path (seconds).

        The handshake RTT is a *two-way* quantity, so a one-way delay of
        ``d`` samples maps to ``rtt = 2 * d / fs`` (PLAN.md §3.6).
        """
        return 2.0 * self.delay_seconds


def find_paths(
    rir: np.ndarray,
    *,
    threshold_db: float = -60.0,
    max_paths: int = 8,
    fs: float = DEFAULT_FS,
) -> list[PathTap]:
    """Estimate the dominant multipath of a room impulse response.

    Normalises ``rir`` to unit peak, keeps the strongest local maxima that are
    at least ``threshold_db`` dB below the peak, and returns them as
    :class:`PathTap` objects sorted by ascending delay (the first tap is the
    earliest / direct path).

    Args:
        rir: Mono room impulse response (any length; 1-D).
        threshold_db: A tap is kept while
            ``20 * log10(|tap| / peak) >= threshold_db`` (i.e. stronger taps
            closer to the peak are retained).
        max_paths: Maximum number of paths to return (the strongest are kept).
        fs: Sampling rate (Hz) for the :class:`PathTap` time conversion.

    Returns:
        A (possibly empty) list of :class:`PathTap`, ordered by delay.
    """
    rir = np.asarray(rir, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(rir)))
    if peak <= 0.0 or not np.isfinite(peak) or max_paths <= 0 or rir.size == 0:
        return []

    envelope = np.abs(rir) / peak
    floor = 10.0 ** (float(threshold_db) / 20.0)

    maxima = _local_maxima(envelope)
    indices = np.nonzero(maxima & (envelope >= floor))[0]
    if indices.size == 0:
        return []

    # Keep the ``max_paths`` strongest, then order them by delay (ascending).
    ranked = sorted(
        zip(indices.tolist(), envelope[indices].tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )[:max_paths]
    chosen = sorted(int(idx) for idx, _ in ranked)
    return [PathTap(delay_samples=i, gain=complex(rir[i] / peak), fs=float(fs)) for i in chosen]


def _local_maxima(env: np.ndarray) -> np.ndarray:
    """Boolean mask of local maxima, including the boundary samples.

    A sample is a maximum when it is >= both neighbours; at the boundaries
    (index 0 / last) it is a maximum when it is >= its single neighbour.  This
    guarantees the global peak is always detected even when it sits at
    sample 0 -- the common case for an RIR whose direct path starts at the
    beginning of the record (so RTT measurement never misses the direct path).
    """
    if env.size == 0:
        return np.zeros(0, dtype=bool)
    if env.size == 1:
        return np.ones(1, dtype=bool)
    out = np.zeros(env.size, dtype=bool)
    out[0] = env[0] >= env[1]
    out[-1] = env[-1] >= env[-2]
    if env.size >= 3:
        out[1:-1] = (env[1:-1] >= env[:-2]) & (env[1:-1] >= env[2:])
    return out


def rake(
    rx: np.ndarray,
    taps: Sequence[PathTap],
    *,
    shift: bool = False,
) -> np.ndarray:
    """Coherent maximum-ratio (RAKE) combining of a received waveform.

    The textbook RAKE output for a known set of paths is the coherent sum of
    each path's delayed copy of the received signal, weighted by that path's
    conjugate gain and normalised by the total collected path power::

        out[n] = sum_k conj(gain_k) * rx[n + d_k] / sum_k |gain_k|^2

    This collects the energy dispersed across the delay spread back onto the
    earliest path, suppressing the inter-path interference that otherwise hurts
    the downstream CSS/MFSK demodulator (the "multipath compression" the sims
    A/B test against the uncombined signal).

    Args:
        rx: Received (noise-contaminated, multipath) waveform.
        taps: Estimated paths from :func:`find_paths`.
        shift: If True, any leading silent samples (before the earliest
            detected path) are trimmed so the combined symbol sits at sample 0;
            if False the full-length output is returned.  The coherent sum is
            aligned to the earliest path either way.

    Returns:
        The combined waveform (a new real ``np.ndarray`` for real input).
    """
    rx = np.asarray(rx, dtype=np.float64).ravel()
    if not taps:
        return rx.copy()

    ordered = sorted(taps, key=lambda t: t.delay_samples)
    size = rx.size
    out = np.zeros(size, dtype=np.complex128)
    for tap in ordered:
        d = tap.delay_samples
        if d >= size:
            continue
        window = rx[d : d + size]  # align this path's copy to sample 0
        out[: window.size] += np.conj(tap.gain) * window
    power = sum(abs(tap.gain) ** 2 for tap in ordered)
    if power > 0.0:
        out = out / power

    result = np.asarray(np.real_if_close(out, tol=1000), dtype=np.float64)
    if shift:
        nz = np.nonzero(np.abs(result) > 1e-9)[0]
        if nz.size:
            start = int(nz[0])
            result = result[start : start + size]
    return result
