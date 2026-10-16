"""S6 — Acoustic link budget: channel capacity versus distance and noise.

This module models the physical-layer budget of an acoustic link under a
free-space (inverse-square) propagation law, and turns the resulting
signal-to-noise ratio into an achievable data rate by delegating to the S3
Shannon primitive :func:`auditory_handshake.metrics.capacity.shannon_capacity`
(single source of truth for the capacity formula — do **not** re-derive
``B * log2(1 + SNR)`` here).

Pipeline (all dB / dB-SPL unless noted)::

    tx_spl_db  --free-space-->  spl_rx(d)  = tx_spl - 20*log10(d / d_ref)
    spl_rx(d) - noise_floor_db = snr(d)    [dB]
    shannon_capacity(bandwidth_hz, snr)    [bit/s]

``noise_floor_db`` and ``tx_spl_db`` are *config-driven*: the simulation
(``sims/01_link_budget.py``) loads them from ``configs/scenarios`` (power-sum
of the noise-source SPL levels) and ``configs/devices`` (``max_spl_db``). The
bandwidth comes from ``configs/defaults.yaml`` (``audio_bands_hz``).

The free-space model is the baseline requested for S6. It does not model room
reverb / image sources, transducer directivity, or distance-dependent noise;
the noise floor is treated as an additive, distance-invariant dB-SPL level.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from auditory_handshake.metrics.capacity import shannon_capacity

__all__ = [
    "REFERENCE_DISTANCE_M",
    "free_space_spl",
    "link_snr",
    "channel_capacity",
    "noise_floor_db",
    "link_budget",
    "link_budget_sweep",
]

#: Default reference distance (metres) at which ``tx_spl_db`` is specified.
REFERENCE_DISTANCE_M = 1.0


def free_space_spl(
    tx_spl_db: float,
    distance_m: float,
    reference_distance_m: float = REFERENCE_DISTANCE_M,
) -> float:
    """Received SPL (dB) from a point source under free-space propagation.

    Uses the inverse-square law: ``spl(d) = tx_spl_db - 20 * log10(d / d_ref)``
    (a 20 dB/decade spherical-spreading loss). Scalar distance in, scalar SPL
    out.

    ``distance_m`` must be ``> 0``.
    """
    d = float(distance_m)
    dref = float(reference_distance_m)
    if d <= 0.0:
        raise ValueError(f"distance_m must be > 0 (got {distance_m})")
    if dref <= 0.0:
        raise ValueError(f"reference_distance_m must be > 0 (got {reference_distance_m})")
    return float(tx_spl_db) - 20.0 * math.log10(d / dref)


def link_snr(spl_rx_db: float, noise_floor_db: float) -> float:
    """Signal-to-noise ratio (dB) at the receiver: ``spl_rx_db - noise_floor_db``."""
    return float(spl_rx_db) - float(noise_floor_db)


def channel_capacity(snr_db: float, bandwidth_hz: float) -> float:
    """Shannon channel capacity (bit/s) for an SNR (dB) and bandwidth (Hz).

    Delegates to the S3 primitive :func:`shannon_capacity`; ``bandwidth_hz``
    must be ``> 0`` (an SNR of ``-inf`` dB or negative is a caller error, see
    ``shannon_capacity``).
    """
    b = float(bandwidth_hz)
    if b <= 0.0:
        raise ValueError(f"bandwidth_hz must be > 0 (got {bandwidth_hz})")
    return shannon_capacity(b, float(snr_db))


def noise_floor_db(levels_db: Sequence[float]) -> float:
    """Combine incoherent noise-source SPL levels (dB) into one noise floor (dB).

    Uncorrelated noise sources add in power, so the combined level is
    ``10 * log10(sum_i 10**(level_i / 10))``. An empty sequence means no
    ambient noise and returns ``0.0`` (dB) (anechoic reference).
    """
    levels = [float(x) for x in levels_db]
    if not levels:
        return 0.0
    return 10.0 * math.log10(sum(10.0 ** (lvl / 10.0) for lvl in levels))


def link_budget(
    distance_m: float,
    tx_spl_db: float,
    noise_floor_db: float,
    bandwidth_hz: float,
) -> dict[str, float]:
    """Full link budget at a single distance.

    Returns a dict with ``distance_m``, ``spl_rx_db``, ``snr_db`` and
    ``capacity_bps`` (bit/s) — the received SPL after free-space spreading, the
    resulting SNR, and the Shannon capacity of that SNR over ``bandwidth_hz``.
    """
    spl = free_space_spl(tx_spl_db, distance_m)
    snr = link_snr(spl, noise_floor_db)
    capacity = channel_capacity(snr, bandwidth_hz)
    return {
        "distance_m": float(distance_m),
        "spl_rx_db": float(spl),
        "snr_db": float(snr),
        "capacity_bps": float(capacity),
    }


def link_budget_sweep(
    distances_m: Sequence[float] | np.ndarray,
    tx_spl_db: float,
    noise_floor_db: float,
    bandwidth_hz: float,
) -> dict[str, np.ndarray]:
    """Link-budget sweep over a grid of distances (for plotting / analysis).

    Returns parallel :class:`numpy.ndarray` columns (``distances_m``,
    ``spl_rx_db``, ``snr_db``, ``capacity_bps``). Capacity is computed by
    delegating to :func:`channel_capacity` (hence :func:`shannon_capacity`)
    per point, keeping the capacity formula in one place.

    Every distance must be ``> 0``.
    """
    d = np.atleast_1d(np.asarray(distances_m, dtype=float))
    if np.any(d <= 0.0):
        raise ValueError(f"distances_m must all be > 0 (got {list(distances_m)})")
    spl_db = np.array([free_space_spl(tx_spl_db, float(x)) for x in d])
    snr_db = spl_db - float(noise_floor_db)
    capacity = np.array([channel_capacity(float(s), bandwidth_hz) for s in snr_db])
    return {
        "distances_m": d,
        "spl_rx_db": spl_db,
        "snr_db": snr_db,
        "capacity_bps": capacity,
    }
