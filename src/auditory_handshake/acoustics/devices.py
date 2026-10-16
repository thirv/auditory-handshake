"""Device band models — loudspeaker and microphone (PLAN.md §2.1, step S4).

PLAN.md §2.1: "Device reality (must be modelled, never assumed flat): small
loudspeakers roll off steeply below ~200–300 Hz; MEMS microphones typically
stop near 16 kHz; both are non-flat and non-linear at high SPL. **The usable
band is a device-limited sub-band**, so band selection is a first-class design
variable."

This module wraps the validated :class:`~auditory_handshake.config.DevicePreset`
(``configs/devices/*.yaml``, S2.6) into a usable model:

- :class:`Transducer` — one device: log-frequency-interpolated response
  (dB re the preset's 0 dB reference), usable band, SPL limit hooks.
- :class:`LinkBand` — the requested band intersected with *both* devices'
  usable bands; per-frequency link gain and the device-aware capacity
  integral.

Capacity convention follows ``metrics.capacity`` (complex baseband, no 1/2
passband factor):

    C = ∫_band log2(1 + SNR_lin · G(f)) df        [bit/s]

with passive link gain ``G(f) = 10**((g_tx(f) + g_rx(f)) / 10)``. A pair of
flat 0 dB devices therefore reduces exactly to ``shannon_capacity``.
Responses are *models, not measurements* (PLAN.md §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from auditory_handshake.config import DevicePreset
from auditory_handshake.metrics.capacity import shannon_capacity

__all__ = ["LinkBand", "Transducer", "device_capacity", "DEFAULT_POINTS"]

#: Deterministic integration grid size (log-spaced) for capacity / mean gain.
DEFAULT_POINTS = 1001


@dataclass(frozen=True)
class Transducer:
    """Band model for one device preset (PLAN.md §2.1).

    The response curve is the preset's dB values, interpolated in **log
    frequency** (transducer curves are standard on log-frequency axes).
    Frequencies outside the preset grid are clamped to the nearest endpoint.
    """

    preset: DevicePreset

    # -- identity / limits ---------------------------------------------------

    @property
    def name(self) -> str:
        return self.preset.name

    @property
    def kind(self) -> str:
        return self.preset.kind

    @property
    def usable_band_hz(self) -> tuple[float, float]:
        """Device-limited usable sub-band (the band selector's search space)."""
        return (self.preset.usable_band_hz[0], self.preset.usable_band_hz[1])

    def in_usable_band(self, f_hz: float) -> bool:
        lo, hi = self.usable_band_hz
        return lo <= f_hz <= hi

    # -- SPL hooks -----------------------------------------------------------

    def headroom_db(self, spl_db: float) -> float:
        """SPL margin to the compress / clip point; negative = over the limit."""
        return self.preset.max_spl_db - spl_db

    def over_spl_limit(self, spl_db: float) -> bool:
        """True when ``spl_db`` exceeds the preset's ``max_spl_db``."""
        return spl_db > self.preset.max_spl_db

    # -- frequency response ----------------------------------------------------

    def response_db(self, f_hz: float) -> float:
        """Transducer response at ``f_hz`` (dB, log-frequency interpolated)."""
        if f_hz <= 0.0:
            msg = f"frequency must be positive, got {f_hz} Hz"
            raise ValueError(msg)
        return cast(float, self._interp(np.log10(f_hz)))

    def response_db_array(self, f_hz: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`response_db` (element-wise, same convention)."""
        if np.any(f_hz <= 0.0):
            msg = "all frequencies must be positive"
            raise ValueError(msg)
        return cast(np.ndarray, self._interp(np.log10(f_hz)))

    def _interp(self, log_f: float | np.ndarray) -> float | np.ndarray:
        return np.interp(
            log_f,
            np.log10(self.preset.frequency_response_hz),
            self.preset.frequency_response_db,
        )


@dataclass(frozen=True)
class LinkBand:
    """Band model of a speaker→mic link over one (device-limited) band.

    ``band_hz`` is the requested band intersected with both devices' usable
    bands, so it is always inside the region where both presets are modelled.
    """

    tx: Transducer
    rx: Transducer
    requested_hz: tuple[float, float]
    band_hz: tuple[float, float]

    @classmethod
    def from_request(cls, tx: Transducer, rx: Transducer, band_hz: tuple[float, float]) -> LinkBand:
        """Intersect ``band_hz`` with both usable bands; fail on no overlap."""
        if band_hz[0] >= band_hz[1]:
            msg = f"band must satisfy f1 < f2, got {band_hz}"
            raise ValueError(msg)
        f1 = max(band_hz[0], tx.usable_band_hz[0], rx.usable_band_hz[0])
        f2 = min(band_hz[1], tx.usable_band_hz[1], rx.usable_band_hz[1])
        if f1 >= f2:
            msg = (
                f"requested band {band_hz} does not overlap the usable range "
                f"[{f1}, {f2}] (tx {tx.name}: {tx.usable_band_hz}, "
                f"rx {rx.name}: {rx.usable_band_hz})"
            )
            raise ValueError(msg)
        return cls(
            tx=tx,
            rx=rx,
            requested_hz=(band_hz[0], band_hz[1]),
            band_hz=(f1, f2),
        )

    def gain_db(self, f_hz: float) -> float:
        """Passive link gain at ``f_hz`` (dB; ≤ 0 for the bundled presets)."""
        return self.tx.response_db(f_hz) + self.rx.response_db(f_hz)

    def gain_db_array(self, f_hz: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`gain_db`."""
        return cast(
            np.ndarray,
            self.tx.response_db_array(f_hz) + self.rx.response_db_array(f_hz),
        )

    def _log_grid(self, n_points: int) -> np.ndarray:
        f1, f2 = self.band_hz
        return np.logspace(np.log10(f1), np.log10(f2), n_points)

    def average_gain_db(self, n_points: int = DEFAULT_POINTS) -> float:
        """Mean link gain (dB) over the band, on a log-uniform grid."""
        return float(np.mean(self.gain_db_array(self._log_grid(n_points))))

    def capacity_bit_s(self, snr_db: float, n_points: int = DEFAULT_POINTS) -> float:
        """Device-aware Shannon capacity over the band (bit/s).

        ``C = ∫ log2(1 + 10**(snr_db/10) · 10**(gain_db(f)/10)) df`` — the
        SNR referenced to the flat-device case, degraded by the passive link
        gain (PLAN.md §2.1). Deterministic trapezoid rule on a log grid.
        """
        grid = self._log_grid(n_points)
        snr_lin = 10.0 ** (snr_db / 10.0)
        gain_lin = 10.0 ** (self.gain_db_array(grid) / 10.0)
        integrand = np.log2(1.0 + snr_lin * gain_lin)
        return float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid)))


def device_capacity(
    tx: Transducer,
    rx: Transducer,
    band_hz: tuple[float, float],
    snr_db: float,
    n_points: int = DEFAULT_POINTS,
) -> float:
    """Device-aware capacity for a requested band (bit/s).

    Convenience wrapper: :meth:`LinkBand.from_request` → :meth:`capacity_bit_s`.
    """
    return LinkBand.from_request(tx, rx, band_hz).capacity_bit_s(snr_db, n_points)


if __name__ == "__main__":
    # S4 checkpoint: the PLAN.md §2.1 bands, now through the device models.
    from auditory_handshake.config import load_device_preset

    tx = Transducer(load_device_preset("small_loudspeaker"))
    rx = Transducer(load_device_preset("mems_microphone"))
    rows = (
        # (band Hz, SNR dB) — PLAN.md §2.1 table order.
        ((300.0, 16_000.0), 30.0),
        ((300.0, 16_000.0), 20.0),
        ((300.0, 16_000.0), 10.0),
        ((300.0, 3_400.0), 20.0),
    )
    print(f"PLAN.md §2.1 — capacity through device band models (tx={tx.name}, rx={rx.name})")
    print(
        " requested band (Hz)  effective band (Hz)  avg gain (dB)  "
        "flat C (kbit/s)  device C (kbit/s)"
    )
    for band, snr in rows:
        link = LinkBand.from_request(tx, rx, band)
        f1, f2 = link.band_hz
        flat = shannon_capacity(f2 - f1, snr) / 1000.0
        dev = link.capacity_bit_s(snr) / 1000.0
        print(
            f"   {f1:8.0f}–{f2:8.0f}   {f1:8.0f}–{f2:8.0f}   "
            f"{link.average_gain_db():11.2f}  {flat:14.1f}  {dev:16.1f}  "
            f"(SNR {snr:.0f} dB)"
        )
