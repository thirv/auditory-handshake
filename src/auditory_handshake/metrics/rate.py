"""Deterministic goodput–audibility trade-off surface (S17.1).

This module defines a parametric design-point grid and the utility / Pareto
frontier logic used to trace the C2 (non-degenerate audibility) criterion
across the trade-off surface.

Key abstractions
----------------
- :class:`DesignPoint` — a frozen, hashable configuration of bands, SNR,
  level, and modulation that fully determines both throughput and audibility.
- :func:`goodput_bps` — effective (BER-penalised, overhead-subtracted) data
  rate for a single design point.
- :func:`aggregate_audibility_cost_db` — Leq-style energy aggregation of the
  per-tone :func:`audibility_cost_db` values across the occupied bands.
- :func:`normalized_utility` — the scalar utility ``U = g̃ − λ · c̃`` used to
  rank design points at a given trade-off weight.
- :func:`nondominated` — extract the Pareto front (maximise goodput,
  minimise cost) from a finite grid.
- :func:`trace_optimum_over_lambda` — walk the utility weight λ and record
  which design point is optimal at each step; the sequence of distinct
  winners *is* the Pareto frontier.
- :func:`c2_verdict` — classify a frontier trace as CORROBORATED (the
  audibility metric discriminates across the front) or DEGENERATE (all front
  points share the same cost → the C2 criterion is not met).

The module is pure (no I/O, no network, no GPU).  Run
``python -m auditory_handshake.metrics.rate`` for a self-check.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .audibility import audibility_cost_db

__all__ = [
    "DesignPoint",
    "LlambdaTrace",
    "gross_bit_rate_bps",
    "goodput_bps",
    "aggregate_audibility_cost_db",
    "make_perceptual_cost_fn",
    "normalized_utility",
    "nondominated",
    "trace_optimum_over_lambda",
    "c2_verdict",
]


# --------------------------------------------------------------------------- #
# Design point
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DesignPoint:
    """A single point on the goodput–audibility trade-off surface.

    Fields
    ------
    n_bands:
        Number of occupied sub-bands (tones) in the transmission.
    bandwidth_per_band_hz:
        Channel bandwidth (Hz) allocated to each sub-band.
    snr_db:
        Per-band signal-to-noise ratio in dB (assumed flat across bands for
        this simplified model).
    level_db:
        Transmit level (dB SPL) at which each tone is emitted.  This is the
        input to :func:`audibility_cost_db`.
    occupied_frequencies_hz:
        Center frequencies (Hz) of the occupied sub-bands.  Length must equal
        ``n_bands``.
    ber:
        Bit-error rate (linear probability, not dB).  Default 1e-6 (a well-
        equalised AWGN channel at moderate SNR).
    overhead_bits:
        Fixed per-frame overhead (handshake, ACK, CRC) subtracted from gross
        throughput to yield goodput.  Default 0 (information-theoretic ceiling).
    """

    n_bands: int
    bandwidth_per_band_hz: float
    snr_db: float
    level_db: float
    occupied_frequencies_hz: tuple[float, ...]
    ber: float = 1e-6
    overhead_bits: float = 0.0

    def __post_init__(self) -> None:
        if self.n_bands < 1:
            raise ValueError(f"n_bands must be >= 1, got {self.n_bands}")
        if len(self.occupied_frequencies_hz) != self.n_bands:
            raise ValueError(
                f"occupied_frequencies_hz length ({len(self.occupied_frequencies_hz)}) "
                f"must equal n_bands ({self.n_bands})"
            )
        if self.bandwidth_per_band_hz <= 0.0:
            raise ValueError(f"bandwidth_per_band_hz must be > 0, got {self.bandwidth_per_band_hz}")
        if self.snr_db <= -100.0:
            raise ValueError(f"snr_db must be > -100 dB, got {self.snr_db}")
        if self.ber < 0.0 or self.ber > 1.0:
            raise ValueError(f"ber must be in [0, 1], got {self.ber}")
        if self.overhead_bits < 0.0:
            raise ValueError(f"overhead_bits must be >= 0, got {self.overhead_bits}")
        for f in self.occupied_frequencies_hz:
            if f <= 0.0:
                raise ValueError(f"occupied frequency must be > 0 Hz, got {f}")


# --------------------------------------------------------------------------- #
# Throughput primitives
# --------------------------------------------------------------------------- #
def gross_bit_rate_bps(p: DesignPoint) -> float:
    """Shannon capacity (bit/s) for the design point.

    Assumes equal per-band SNR:  ``R = n_bands · B · log2(1 + SNR_linear)``.
    """
    snr_linear = 10.0 ** (p.snr_db / 10.0)
    return p.n_bands * p.bandwidth_per_band_hz * math.log2(1.0 + snr_linear)


def goodput_bps(p: DesignPoint) -> float:
    """Effective data rate after BER losses and fixed overhead (bit/s).

    ``goodput = max(0, gross · (1 − ber) − overhead_bits)``
    """
    gross = gross_bit_rate_bps(p)
    return max(0.0, gross * (1.0 - p.ber) - p.overhead_bits)


# --------------------------------------------------------------------------- #
# Audibility aggregation
# --------------------------------------------------------------------------- #
def aggregate_audibility_cost_db(p: DesignPoint) -> float:
    """Total perceived audibility cost (dB) across all occupied bands.

    Aggregates per-tone :func:`audibility_cost_db` values using the standard
    energy (Leq) combination rule::

        L_total = 10 · log10( Σ 10^(L_i / 10) )

    This is the physically meaningful measure of total sound energy and
    scales with the number of occupied bands, creating the correct trade-off
    with goodput (which also scales with band count).
    """
    costs = np.array(
        [audibility_cost_db(f, p.level_db) for f in p.occupied_frequencies_hz],
        dtype=np.float64,
    )
    # Numerically-stable energy sum: factor out the max to avoid overflow.
    c_max = float(costs.max())
    total = c_max + 10.0 * float(np.log10(np.sum(10.0 ** ((costs - c_max) / 10.0))))
    return total


def make_perceptual_cost_fn(
    *,
    fs: int = 16000,
    duration_s: float = 4.0,
    signal_type: str = "noise",
    noise_kind: str = "broadband",
) -> Callable[[DesignPoint], float]:
    """Build a C2 *perceptual* cost function: cost = perceived annoyance.

    For each :class:`DesignPoint`, its occupied bands are rendered as broadband
    noise (one :class:`~auditory_handshake.config.NoiseSource` per band, each at
    the point's transmit level) via :func:`synthesize_noise_floor`, and reduced
    to a single scalar by :func:`perceptual_annoyance` (loudness in
    sones × a 0–1 penalty, or a frequency-dependent modulation factor for
    ``css``/``mfsk``).  This is the C2 cost that replaces the parametric dB
    audibility term: it reflects how *annoying the transmitted sound is
    perceived to be*, not merely how loud it is.  For ``noise``/``sharp``/
    ``rough`` the value satisfies ``0 <= annoyance <= N_avg``; for
    ``css``/``mfsk`` it is ``N_avg`` times a factor ``>= 1`` (unbounded above,
    growing with sharpness/roughness).  A point with no occupied bands costs
    ``0.0``.

    The perceptual backend (``mosqito``) and the noise synthesiser are imported
    lazily so :mod:`rate` stays light and the legacy dB path pays for neither.

    Parameters
    ----------
    fs:
        Sampling rate (Hz) of the synthesised signal.
    duration_s:
        Duration (s) of the synthesised signal.  The signal must be long enough
        for the perceptual model (``duration_s * fs`` >= 48 000 samples); the
        defaults give 64 000 samples at 16 kHz.
    signal_type:
        Passed through to :func:`perceptual_annoyance` (``"noise"`` default).
        Use ``"css"`` / ``"mfsk"`` for modulated-carrier cost models, which
        scale the annoyance by a frequency-dependent sharpness/roughness
        factor.
    noise_kind:
        Passed through to :func:`perceptual_annoyance` (``"broadband"``
        default); only used by the ``"noise"`` signal type.
    """
    from auditory_handshake.acoustics.noise import synthesize_noise_floor
    from auditory_handshake.config import NoiseSource
    from auditory_handshake.metrics.perceptual import perceptual_annoyance

    def cost(p: DesignPoint) -> float:
        if p.n_bands <= 0:
            return 0.0
        # One incoherent band source per occupied centre, at the design point's
        # transmit level.  ``rotor`` is the only kind that honours ``center_hz``
        # (a tone stack at that frequency), so it is the natural carrier for a
        # band-selective transmit spectrum; the incoherent sum of the per-band
        # sources reproduces the total transmit SPL (powers add, levels don't).
        centers = [f for f in p.occupied_frequencies_hz if 0.0 < f < fs / 2.0]
        if not centers:
            return 0.0
        sources = [
            NoiseSource(
                kind="rotor",
                level_db_spl=p.level_db,
                distance_m=1.0,
                center_hz=f,
            )
            for f in centers
        ]
        x = synthesize_noise_floor(sources, duration_s, fs)
        return float(perceptual_annoyance(x, fs, signal_type, noise_kind).annoyance)

    return cost


# --------------------------------------------------------------------------- #
# Utility and frontier
# --------------------------------------------------------------------------- #
def normalized_utility(
    p: DesignPoint,
    lambda_: float,
    *,
    goodput_scale: float,
    cost_scale: float,
    cost_fn: Callable[[DesignPoint], float] | None = None,
) -> float:
    """Scalar utility  ``U = g̃ − λ · c̃``  (dimensionless).

    Parameters
    ----------
    p:
        The design point to score.
    lambda_:
        Trade-off weight (>= 0).  Larger values penalise audibility more.
    goodput_scale:
        Normalisation reference for goodput (typically the grid maximum).
    cost_scale:
        Normalisation reference for audibility cost (typically the grid
        maximum).

    Raises
    ------
    ValueError
        If ``lambda_ < 0`` or either scale is non-positive.
    """
    if lambda_ < 0.0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}")
    if goodput_scale <= 0.0:
        raise ValueError(f"goodput_scale must be > 0, got {goodput_scale}")
    if cost_scale <= 0.0:
        raise ValueError(f"cost_scale must be > 0, got {cost_scale}")
    g = goodput_bps(p) / goodput_scale
    c = (cost_fn or aggregate_audibility_cost_db)(p) / cost_scale
    return g - lambda_ * c


def nondominated(
    points: Sequence[DesignPoint],
    *,
    cost_fn: Callable[[DesignPoint], float] | None = None,
) -> list[DesignPoint]:
    """Return the Pareto-optimal subset (maximise goodput, minimise cost).

    A point is *dominated* if another point has both ``goodput >=`` and
    ``cost <=`` with at least one strict inequality.  The Pareto front is
    the set of non-dominated points, returned in their original order.
    """
    n = len(points)
    if n == 0:
        return []
    cost = cost_fn or aggregate_audibility_cost_db
    scores: list[tuple[float, float]] = [(goodput_bps(p), cost(p)) for p in points]
    is_dominated = [False] * n
    for i in range(n):
        gi, ci = scores[i]
        for j in range(n):
            if i == j:
                continue
            gj, cj = scores[j]
            if gj >= gi and cj <= ci and (gj > gi or cj < ci):
                is_dominated[i] = True
                break
    return [points[i] for i in range(n) if not is_dominated[i]]


# --------------------------------------------------------------------------- #
# Frontier trace
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LlambdaTrace:
    """Result of tracing the optimal design point as λ varies.

    Fields
    ------
    lambdas:
        The sequence of trade-off weights that were swept.
    frontier_points:
        The design point selected as optimal at each λ (may repeat if the
        same point wins for a range of λ).
    frontier_goodput_bps:
        Goodput (bit/s) at each frontier point.
    frontier_cost_db:
        Aggregate audibility cost (dB) at each frontier point.
    n_distinct:
        Number of *distinct* design points on the frontier (the size of the
        true Pareto front as revealed by this trace).
    """

    lambdas: tuple[float, ...]
    frontier_points: tuple[DesignPoint, ...]
    frontier_goodput_bps: tuple[float, ...]
    frontier_cost_db: tuple[float, ...]
    n_distinct: int


def trace_optimum_over_lambda(
    points: Sequence[DesignPoint],
    lambdas: Sequence[float],
    *,
    cost_fn: Callable[[DesignPoint], float] | None = None,
) -> LlambdaTrace:
    """Sweep λ and record the optimal design point at each value.

    Normalisation references (goodput_scale, cost_scale) are the grid
    maxima, so the utility is dimensionless and comparable across the grid.

    Parameters
    ----------
    points:
        Finite grid of design points to choose from.
    lambdas:
        Sequence of trade-off weights to sweep (typically increasing from 0
        toward large values, so the frontier is traced from the goodput
        extreme to the audibility extreme).

    Returns
    -------
    LlambdaTrace

    Raises
    ------
    ValueError
        If ``points`` is empty or all goodput/cost values are zero.
    """
    if not points:
        raise ValueError("points must be non-empty")

    goodputs = np.array([goodput_bps(p) for p in points], dtype=np.float64)
    cost = cost_fn or aggregate_audibility_cost_db
    costs = np.array([cost(p) for p in points], dtype=np.float64)
    goodput_scale = float(goodputs.max())
    cost_scale = float(costs.max())
    if goodput_scale <= 0.0:
        raise ValueError("all goodput values are zero; cannot normalise")
    if cost_scale <= 0.0:
        raise ValueError("all cost values are zero; cannot normalise")

    lambdas_t = tuple(float(lm) for lm in lambdas)
    best_indices: list[int] = []
    for lam in lambdas_t:
        utilities = goodputs / goodput_scale - lam * costs / cost_scale
        best_indices.append(int(np.argmax(utilities)))

    frontier_points = tuple(points[i] for i in best_indices)
    frontier_goodput = tuple(float(goodputs[i]) for i in best_indices)
    frontier_cost = tuple(float(costs[i]) for i in best_indices)
    n_distinct = len({id(p) for p in frontier_points})

    return LlambdaTrace(
        lambdas=lambdas_t,
        frontier_points=frontier_points,
        frontier_goodput_bps=frontier_goodput,
        frontier_cost_db=frontier_cost,
        n_distinct=n_distinct,
    )


# --------------------------------------------------------------------------- #
# C2 verdict
# --------------------------------------------------------------------------- #
def c2_verdict(trace: LlambdaTrace) -> str:
    """Classify a frontier trace per the C2 non-degeneracy criterion.

    Returns
    -------
    str
        ``"CORROBORATED"`` if the frontier contains at least two points with
        distinct aggregate audibility costs (the metric discriminates, the
        trade-off surface is meaningful).

        ``"DEGENERATE"`` if all frontier points share the same cost (the
        audibility metric fails to differentiate — C2 is violated).
    """
    distinct_costs = len({round(c, 6) for c in trace.frontier_cost_db})
    return "CORROBORATED" if distinct_costs >= 2 else "DEGENERATE"


# --------------------------------------------------------------------------- #
# Self-check (``python -m auditory_handshake.metrics.rate``)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    """Build a small grid and print the frontier trace + verdict."""
    print("S17.1 — goodput vs. audibility trade-off (self-check)")
    print("=" * 70)

    base_f = (500.0, 1000.0, 2000.0, 4000.0)
    points: list[DesignPoint] = []
    for n_bands in (1, 2, 3, 4):
        for level in (50.0, 60.0, 70.0):
            pts = base_f[:n_bands]
            points.append(
                DesignPoint(
                    n_bands=n_bands,
                    bandwidth_per_band_hz=5000.0,
                    snr_db=20.0,
                    level_db=level,
                    occupied_frequencies_hz=pts,
                )
            )

    print(f"Grid size: {len(points)} design points")
    front = nondominated(points)
    print(f"Pareto front: {len(front)} nondominated points")
    for p in front:
        g = goodput_bps(p)
        c = aggregate_audibility_cost_db(p)
        print(
            f"  n_bands={p.n_bands} level={p.level_db:.0f} dB  goodput={g:,.0f} bit/s  cost={c:.2f} dB"
        )

    lambdas = np.geomspace(0.01, 100.0, 30).tolist()
    trace = trace_optimum_over_lambda(points, lambdas)
    verdict = c2_verdict(trace)
    print(f"\nFrontier trace: {len(trace.lambdas)} λ values, {trace.n_distinct} distinct winners")
    print(f"C2 verdict: {verdict}")

    if verdict == "CORROBORATED":
        print("  → audibility metric discriminates across the front (C2 met)")
    else:
        print("  → audibility metric is degenerate (C2 NOT met)")


if __name__ == "__main__":  # pragma: no cover - checkpoint helper
    _demo()
