"""Channel capacity (S3): Shannon bound and water-filling allocation.

Implements the §2.1 "bandwidth ceiling" of PLAN.md:

- :func:`shannon_capacity` — the flat-noise Shannon bound ``C = B · log2(1 + SNR)``.
- :func:`waterfilling_capacity` — optimal power allocation over non-flat
  sub-bands (water-filling), the theoretical ceiling a PHY should approach.

Convention: complex baseband, so capacity per sub-band of width ``Δf`` is
``Δf · log2(1 + SNR_i)`` (no 1/2 passband factor). All bandwidths in Hz,
capacities in bit/s.

Run ``python -m auditory_handshake.metrics.capacity`` to print the PLAN.md
§2.1 capacity table (S3 checkpoint).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "WaterfillResult",
    "shannon_capacity",
    "waterfilling_capacity",
]


def shannon_capacity(bandwidth_hz: float, snr_db: float) -> float:
    """Shannon capacity ``C = B · log2(1 + SNR)`` in bit/s (PLAN.md §2.1).

    Parameters
    ----------
    bandwidth_hz:
        Usable one-sided bandwidth in Hz (must be > 0).
    snr_db:
        Signal-to-noise ratio in dB over the band.
    """
    if bandwidth_hz <= 0:
        raise ValueError(f"bandwidth_hz must be > 0, got {bandwidth_hz}")
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(bandwidth_hz * np.log2(1.0 + snr_linear))


def _water_level(normalized_noise: NDArray[np.float64], total_power: float) -> float:
    """Solve ``sum(max(P0 - N_i, 0)) = total_power`` for the water level P0.

    ``f(P0)`` is monotone non-decreasing in P0, so bisection is exact and
    dependency-free.
    """
    lo = 0.0
    hi = float(normalized_noise.max()) + total_power  # all sub-bands active
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.maximum(mid - normalized_noise, 0.0))) > total_power:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class WaterfillResult:
    """Result of a water-filling power allocation."""

    capacity_bits_s: float
    allocated_power: NDArray[np.float64]
    water_level: float
    active_subbands: int


def waterfilling_capacity(
    noise_powers: Sequence[float],
    channel_gains: Sequence[float] | None = None,
    total_power: float = 1.0,
    subband_width_hz: float = 1.0,
) -> WaterfillResult:
    """Water-filling capacity over ``len(noise_powers)`` sub-bands.

    Allocates ``total_power`` across sub-bands of width ``subband_width_hz``
    against per-sub-band noise powers ``noise_powers`` (optionally shaped by
    ``channel_gains``), following ``P_i = max(P0 - N_i / |H_i|^2, 0)`` where
    the water level ``P0`` satisfies ``sum(P_i) = total_power``.

    With flat noise and unity gains this reduces to
    :func:`shannon_capacity` for the same total SNR.
    """
    noise = np.asarray(noise_powers, dtype=np.float64)
    if noise.ndim != 1 or noise.size == 0:
        raise ValueError("noise_powers must be a non-empty 1-D sequence")
    if np.any(noise <= 0):
        raise ValueError("noise_powers must all be > 0")
    if channel_gains is None:
        gains = np.ones_like(noise)
    else:
        gains = np.asarray(channel_gains, dtype=np.float64)
        if gains.shape != noise.shape:
            raise ValueError("channel_gains must match noise_powers length")
        if np.any(gains <= 0):
            raise ValueError("channel_gains must all be > 0")
    if total_power < 0:
        raise ValueError(f"total_power must be >= 0, got {total_power}")
    if subband_width_hz <= 0:
        raise ValueError(f"subband_width_hz must be > 0, got {subband_width_hz}")

    normalized = noise / (gains**2)
    p0 = _water_level(normalized, total_power)
    power = np.maximum(p0 - normalized, 0.0)
    active = power > 0
    capacity = float(subband_width_hz * np.sum(np.log2(1.0 + power[active] / noise[active])))
    return WaterfillResult(
        capacity_bits_s=capacity,
        allocated_power=power,
        water_level=float(p0),
        active_subbands=int(active.sum()),
    )


if __name__ == "__main__":
    print("PLAN.md §2.1 — bandwidth ceiling, C = B · log2(1 + SNR)")
    print(f"{'band (Hz)':>10} {'SNR (dB)':>9} {'capacity (kbit/s)':>18}")
    for b, s in [(15_000, 30), (15_000, 20), (15_000, 10), (4_000, 20)]:
        print(f"{b:>10} {s:>9} {shannon_capacity(b, s) / 1000:>18.1f}")
