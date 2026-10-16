"""C1 / TTCH model and RT60-sensitivity falsification (PLAN.md §2.4, §1.4, M6).

The C1 crux (PLAN.md §1.2) is that the acoustic handshake **must work in the
noisiest, most reverberant room**, so its turnaround time (TTCH) must *scale*
with the room's reverberation time (RT60). PLAN.md §1.4 makes this a falsifiable
criterion:

    "C1 dies if RTT is statistically insensitive — or worse, anti-correlated —
     to RT60."

This module provides the three pure building blocks used to check that, without
any I/O:

- :func:`ttch_rtt_s` — the PLAN.md §2.4 round-trip turnaround model
  ``TTCH_RTT = 2·(T_preamble + T_frame + k·RT60) + T_proc`` (a *design premise*
  that is monotone in RT60 for ``k > 0``).
- :func:`regress_rt60_sensitivity` — the PLAN.md §1.4 falsification test: an
  ordinary-least-squares fit of a *measured* per-room latency on RT60, with a
  Pearson ``r`` and ``t``-statistic and a plain-English verdict.
- :func:`classify_regime` — the PLAN.md §2.4 / §3.1 win/lose split
  (``RT60 < 0.35 s`` → "win": sub-meter / cabin / docking; else → "lose":
  reverberant hall).

The round-trip model uses the PLAN.md §2.4 echo-tail constant ``k ∈ [0.2, 0.5]``
so that ``T_echo_tail = k·RT60``. The 2-way multiplier is the number of transmit
turnarounds; PLAN.md §2.4 gives 2 for a 2-way handshake. All times are in
seconds; callers pass measured quantities (e.g. the M4 Channel-arm one-way
preamble latency) for the *falsification* half and design values for the *model*
half.

Run ``python -m auditory_handshake.metrics.ttch`` to print a small PLAN.md §2.4
TTCH table and the §3.1 regime split (checkpoint helper, like ``capacity.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RT60Sensitivity",
    "ttch_rtt_s",
    "regress_rt60_sensitivity",
    "classify_regime",
]

# PLAN.md §2.4 defaults (documented assumptions; overridable per call).
DEFAULT_K = 0.35  # mid of the PLAN.md §2.4 echo-tail range [0.2, 0.5]
DEFAULT_T_PROC_S = 0.010  # fixed per-turnaround processing/codec estimate (T_proc)
DEFAULT_TURNAROUNDS = 2  # PLAN.md §2.4: 2-way handshake = 2 turnarounds
REGIME_THRESHOLD_S = 0.35  # PLAN.md §2.4 / §3.1: win (<) vs lose (>=)


def ttch_rtt_s(
    rt60_s: float,
    t_preamble_s: float,
    t_frame_s: float,
    t_proc_s: float = DEFAULT_T_PROC_S,
    k: float = DEFAULT_K,
    turnarounds: int | float = DEFAULT_TURNAROUNDS,
) -> float:
    """Round-trip turnaround time, in seconds (PLAN.md §2.4).

    ``TTCH_RTT = turnarounds · (T_preamble + T_frame + k·RT60) + T_proc``

    where ``T_echo_tail = k·RT60`` (echo/decay tail; ``k ∈ [0.2, 0.5]`` per
    PLAN.md §2.4) and ``turnarounds`` is the number of transmit turnarounds
    (2 for a 2-way handshake, PLAN.md §2.4).

    Parameters
    ----------
    rt60_s:
        Room reverberation time in seconds (must be ``>= 0``).
    t_preamble_s:
        Preamble detection time in seconds (must be ``>= 0``).
    t_frame_s:
        Frame transmit time in seconds (must be ``>= 0``).
    t_proc_s:
        Fixed per-turnaround processing/codec time in seconds (default 10 ms).
    k:
        Echo-tail constant (must be ``> 0``); PLAN.md §2.4 suggests ``[0.2, 0.5]``.
    turnarounds:
        Number of transmit turnarounds (must be ``>= 1``); 2 = 2-way handshake.
        May be fractional when it is an *expected* value, e.g. the expected
        clean-slot count from a contention model (see
        :mod:`auditory_handshake.contention`); the model is affine in it.

    Returns
    -------
    float
        ``TTCH_RTT`` in seconds.
    """
    for name, value in (
        ("rt60_s", rt60_s),
        ("t_preamble_s", t_preamble_s),
        ("t_frame_s", t_frame_s),
        ("t_proc_s", t_proc_s),
    ):
        if not (value >= 0.0):
            raise ValueError(f"{name} must be >= 0, got {value}")
    if k <= 0.0:
        raise ValueError(f"k must be > 0, got {k} (PLAN.md §2.4: k ∈ [0.2, 0.5])")
    if turnarounds < 1:
        raise ValueError(f"turnarounds must be >= 1, got {turnarounds}")
    t_echo_tail_s = k * rt60_s
    return float(turnarounds * (t_preamble_s + t_frame_s + t_echo_tail_s) + t_proc_s)


def classify_regime(rt60_s: float, threshold_s: float = REGIME_THRESHOLD_S) -> str:
    """Win/lose regime for a room (PLAN.md §2.4 / §3.1).

    ``RT60 < threshold_s`` → ``"win"`` (sub-meter / cabin / docking — C1 is meant
    to win here); otherwise → ``"lose"`` (reverberant hall — where C1 must still
    work but is expected to be slower).
    """
    if not (rt60_s >= 0.0):
        raise ValueError(f"rt60_s must be >= 0, got {rt60_s}")
    if not (threshold_s > 0.0):
        raise ValueError(f"threshold_s must be > 0, got {threshold_s}")
    return "win" if rt60_s < threshold_s else "lose"


@dataclass(frozen=True)
class RT60Sensitivity:
    """Result of :func:`regress_rt60_sensitivity` (PLAN.md §1.4 falsification).

    Attributes
    ----------
    n:
        Number of (RT60, latency) samples used.
    slope:
        OLS slope of ``latency_s`` on ``rt60_s`` (s latency per s RT60).
    intercept:
        OLS intercept of that fit (s latency).
    pearson_r:
        Pearson correlation of the two series, in ``[-1, 1]``.
    t_stat:
        ``t``-statistic for ``slope == 0`` (``None`` when ``n < 3`` or zero
        residual variance makes it undefined).
    verdict:
        ``"INCONCLUSIVE"`` (``n < 3``), ``"CONTRADICTED"`` (slope ``<= 0``),
        ``"AT_RISK"`` (weakly positive), ``"WEAKLY_CORROBORATED"``, or
        ``"CORROBORATED"`` (clearly positive and significant) — PLAN.md §1.4
        semantics.
    """

    n: int
    slope: float
    intercept: float
    pearson_r: float
    t_stat: float | None
    verdict: str


def regress_rt60_sensitivity(
    rt60_s: Sequence[float],
    latency_s: Sequence[float],
    weak_t: float = 2.0,
    strong_t: float = 3.0,
) -> RT60Sensitivity:
    """Falsification test: is a measured latency statistically sensitive to RT60?

    Fits ``latency_s = intercept + slope · rt60_s`` by ordinary least squares and
    reports the slope, Pearson ``r``, and a ``t``-statistic for ``slope == 0``.
    This is the PLAN.md §1.4 check — C1 dies when the slope is not clearly
    positive.

    Parameters
    ----------
    rt60_s:
        Per-room reverberation times in seconds (1-D, must have non-zero variance).
    latency_s:
        Per-room measured latencies in seconds, aligned with ``rt60_s``.
    weak_t / strong_t:
        ``|t``-statistic thresholds separating ``AT_RISK`` /
        ``WEAKLY_CORROBORATED`` / ``CORROBORATED`` for a positive slope.

    Returns
    -------
    RT60Sensitivity
    """
    x = np.asarray(rt60_s, dtype=float)
    y = np.asarray(latency_s, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("rt60_s and latency_s must be 1-D")
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    n = int(x.size)
    if n < 2:
        raise ValueError(f"need >= 2 samples, got {n}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("non-finite values in input")

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0.0:
        raise ValueError("rt60_s values must not all be identical (zero variance)")
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot > 0.0:
        pearson_r = math.copysign(math.sqrt(max(0.0, 1.0 - ss_res / ss_tot)), slope)
    else:
        pearson_r = 0.0

    if n > 2:
        mse = ss_res / (n - 2)
        if mse > 0.0:
            t_stat: float | None = slope / math.sqrt(mse / sxx)
        else:
            t_stat = math.copysign(float("inf"), slope)
    else:
        t_stat = None

    if n < 3:
        verdict = "INCONCLUSIVE"
    elif slope <= 0.0:
        verdict = "CONTRADICTED"
    elif t_stat is None:
        verdict = "AT_RISK"
    elif abs(t_stat) >= strong_t:
        verdict = "CORROBORATED"
    elif abs(t_stat) >= weak_t:
        verdict = "WEAKLY_CORROBORATED"
    else:
        verdict = "AT_RISK"

    return RT60Sensitivity(
        n=n,
        slope=float(slope),
        intercept=float(intercept),
        pearson_r=float(pearson_r),
        t_stat=t_stat,
        verdict=verdict,
    )


if __name__ == "__main__":  # pragma: no cover - checkpoint helper
    # PLAN.md §2.4 TTCH table: nominal 1-way preamble latency 1.25 ms, 32-bit
    # control frame at 1024 bit/s (31.25 ms), T_proc 10 ms.
    t_preamble_s = 0.00125
    t_frame_s = 32.0 / 1024.0
    print("PLAN.md §2.4 TTCH model (2-way handshake):")
    print(f"  T_frame = {t_frame_s * 1000.0:.2f} ms (32-bit control @ 1024 bit/s), T_proc = 10 ms")
    print(f"  {'RT60 (s)':>9} {'k=0.2 (ms)':>12} {'k=0.35 (ms)':>12} {'k=0.5 (ms)':>12}")
    for rt60 in (0.0, 0.102, 0.247, 0.35, 0.5, 0.711, 1.0):
        row = [ttch_rtt_s(rt60, t_preamble_s, t_frame_s, k=k) * 1000.0 for k in (0.2, 0.35, 0.5)]
        print(f"  {rt60:9.3f} {row[0]:12.1f} {row[1]:12.1f} {row[2]:12.1f}")
    print("\nPLAN.md §3.1 regime split (RT60 < 0.35 s = 'win'):")
    for name, rt60 in (
        ("cabin", 0.102),
        ("corridor", 0.247),
        ("small_office", 0.425),
        ("small_hall", 0.711),
    ):
        print(f"  {name:>14}: RT60={rt60:.3f} s -> {classify_regime(rt60)}")
