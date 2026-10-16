"""Tests for the design-point → modem mapping helpers (``phy/design.py``).

Pure unit tests (no I/O, no GPU).  They pin down:

- :func:`design_to_mfsk` / :func:`design_to_css` produce a config whose
  ``fs_hz`` is the lowest occupied band and whose spacing matches the
  median band gap (round-trip).
- The strict band floor/ceiling and constellation-size validation: an
  infeasible design point raises :class:`DesignInfeasibleError` (and
  :func:`band_feasible` reports ``False``) for the *correct* reason.
- :func:`measure_design_bler` returns a zero-error result on a clean channel
  and raises :class:`DesignInfeasibleError` (before simulating) for an
  infeasible design point.
"""

from __future__ import annotations

import pytest

from auditory_handshake.metrics.rate import DesignPoint
from auditory_handshake.phy.design import (
    DesignInfeasibleError,
    band_feasible,
    design_to_css,
    design_to_mfsk,
    measure_design_bler,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mfsk_point(base: float = 8000.0, spacing: float = 1000.0) -> DesignPoint:
    """A 4-band design point with uniform ``spacing`` starting at ``base``."""
    freqs = tuple(base + k * spacing for k in range(4))
    return DesignPoint(
        n_bands=4,
        bandwidth_per_band_hz=spacing,
        snr_db=20.0,
        level_db=60.0,
        occupied_frequencies_hz=freqs,
        ber=1e-6,
        overhead_bits=0.0,
    )


def _css_point(base: float = 10_000.0, spacing: float = 500.0) -> DesignPoint:
    """A 4-chirp design point with uniform ``spacing`` starting at ``base``."""
    freqs = tuple(base + k * spacing for k in range(4))
    return DesignPoint(
        n_bands=4,
        bandwidth_per_band_hz=spacing,
        snr_db=20.0,
        level_db=60.0,
        occupied_frequencies_hz=freqs,
        ber=1e-6,
        overhead_bits=0.0,
    )


# --------------------------------------------------------------------------- #
# design_to_mfsk — round-trip
# --------------------------------------------------------------------------- #
def test_design_to_mfsk_round_trip() -> None:
    design = _mfsk_point(base=8000.0, spacing=1000.0)
    cfg = design_to_mfsk(design, band_lo=8000.0, band_hi=12000.0)

    assert cfg.fs_hz == pytest.approx(8000.0)
    assert cfg.symbols_per_second == 1000
    # MFSK tone lattice: fs_hz + k * symbols_per_second for k in 0..3.
    top = cfg.fs_hz + 3 * cfg.symbols_per_second
    assert top == pytest.approx(11000.0)
    assert cfg.fs_hz >= 8000.0
    assert top <= 12000.0


def test_design_to_mfsk_defaults_from_config() -> None:
    # Default band [8000, 12000] from configs/defaults.yaml accepts the same
    # design point without explicit overrides.
    cfg = design_to_mfsk(_mfsk_point())
    assert cfg.fs_hz == pytest.approx(8000.0)
    assert cfg.symbols_per_second == 1000


# --------------------------------------------------------------------------- #
# design_to_css — round-trip
# --------------------------------------------------------------------------- #
def test_design_to_css_round_trip() -> None:
    design = _css_point(base=10_000.0, spacing=500.0)
    cfg = design_to_css(design, band_lo=10_000.0, band_hi=20_000.0)

    assert cfg.fs_hz == pytest.approx(10_000.0)
    assert cfg.n_chirps == 4
    # Chirp m starts at fs_hz + m * chirp_spacing; chirp_spacing = total_bw / n.
    assert cfg.total_bandwidth == pytest.approx(4 * 500.0)
    assert cfg.chirp_spacing == pytest.approx(500.0)
    top = cfg.fs_hz + (cfg.n_chirps - 1) * cfg.chirp_spacing + cfg.total_bandwidth
    assert top == pytest.approx(13_500.0)
    assert cfg.fs_hz >= 10_000.0
    assert top <= 20_000.0


# --------------------------------------------------------------------------- #
# Infeasible design points (strict validation)
# --------------------------------------------------------------------------- #
def test_mfsk_band_floor_infeasible() -> None:
    # base 5000 Hz < mfsk band floor 8000 Hz.
    design = _mfsk_point(base=5000.0, spacing=1000.0)
    with pytest.raises(DesignInfeasibleError, match="band floor"):
        design_to_mfsk(design, band_lo=8000.0, band_hi=12000.0)


def test_mfsk_band_ceiling_infeasible() -> None:
    # base 11000 Hz, spacing 1000 -> top tone 14000 Hz > ceiling 12000 Hz.
    design = _mfsk_point(base=11000.0, spacing=1000.0)
    with pytest.raises(DesignInfeasibleError, match="band ceiling"):
        design_to_mfsk(design, band_lo=8000.0, band_hi=12000.0)


def test_mfsk_constellation_infeasible() -> None:
    # MFSK is a fixed 4-tone modem: a 2-band design point cannot map 1:1.
    design = DesignPoint(
        n_bands=2,
        bandwidth_per_band_hz=1000.0,
        snr_db=20.0,
        level_db=60.0,
        occupied_frequencies_hz=(8000.0, 9000.0),
        ber=1e-6,
        overhead_bits=0.0,
    )
    with pytest.raises(DesignInfeasibleError, match="fixed 4-tone"):
        design_to_mfsk(design, band_lo=8000.0, band_hi=12000.0)


def test_css_constellation_infeasible() -> None:
    # CSS chirp count must be a power of two >= 2; 3 is not.
    design = DesignPoint(
        n_bands=3,
        bandwidth_per_band_hz=500.0,
        snr_db=20.0,
        level_db=60.0,
        occupied_frequencies_hz=(10_000.0, 10_500.0, 11_000.0),
        ber=1e-6,
        overhead_bits=0.0,
    )
    with pytest.raises(DesignInfeasibleError, match="power of two"):
        design_to_css(design, band_lo=10_000.0, band_hi=20_000.0)


def test_css_band_ceiling_infeasible() -> None:
    # base 10000, spacing 2000, n=4 -> top chirp edge 24000 Hz >= Nyquist and
    # > ceiling 20000 Hz.  Either constraint trips the mapping.
    design = _css_point(base=10_000.0, spacing=2000.0)
    with pytest.raises(DesignInfeasibleError):
        design_to_css(design, band_lo=10_000.0, band_hi=20_000.0)


# --------------------------------------------------------------------------- #
# band_feasible — boolean mirror of the strict mapping checks
# --------------------------------------------------------------------------- #
def test_band_feasible_mfsk() -> None:
    assert band_feasible(_mfsk_point(base=8000.0, spacing=1000.0), 8000.0, 12000.0)
    # Below the floor.
    assert not band_feasible(_mfsk_point(base=5000.0), 8000.0, 12000.0)
    # Above the ceiling.
    assert not band_feasible(_mfsk_point(base=11000.0), 8000.0, 12000.0)


def test_band_feasible_css() -> None:
    assert band_feasible(_css_point(base=10_000.0, spacing=500.0), 10_000.0, 20_000.0)
    # 3-chirp constellation is infeasible for CSS.
    design = DesignPoint(
        n_bands=3,
        bandwidth_per_band_hz=500.0,
        snr_db=20.0,
        level_db=60.0,
        occupied_frequencies_hz=(10_000.0, 10_500.0, 11_000.0),
        ber=1e-6,
        overhead_bits=0.0,
    )
    assert not band_feasible(design, 10_000.0, 20_000.0, mode="css")


# --------------------------------------------------------------------------- #
# measure_design_bler
# --------------------------------------------------------------------------- #
def test_measure_design_bler_clean_channel_mfsk() -> None:
    result = measure_design_bler(
        _mfsk_point(base=8000.0, spacing=1000.0),
        mode="mfsk",
        num_symbols=16,
        seed=123,
        band_lo=8000.0,
        band_hi=12000.0,
    )
    assert result.mode == "mfsk"
    assert result.snr_db is None
    assert result.num_symbols == 16
    assert result.ber == 0.0
    assert result.bler == 0.0
    assert result.goodput_bps >= 0.0
    assert result.audibility_cost_db > 0.0


def test_measure_design_bler_clean_channel_css() -> None:
    result = measure_design_bler(
        _css_point(base=10_000.0, spacing=500.0),
        mode="css",
        num_symbols=8,
        seed=123,
        band_lo=10_000.0,
        band_hi=20_000.0,
    )
    assert result.mode == "css"
    assert result.ber == 0.0
    assert result.bler == 0.0


def test_measure_design_bler_infeasible_raises() -> None:
    design = _mfsk_point(base=5000.0, spacing=1000.0)  # below the mfsk band floor
    with pytest.raises(DesignInfeasibleError):
        measure_design_bler(design, mode="mfsk", num_symbols=16, band_lo=8000.0, band_hi=12000.0)


def test_measure_design_bler_rejects_bad_symbol_count() -> None:
    with pytest.raises(ValueError, match="num_symbols"):
        measure_design_bler(_mfsk_point(), mode="mfsk", num_symbols=0)
