"""Tests for S11.1 noise synthesis (:mod:`auditory_handshake.acoustics.noise`).

Conventions follow the S10.2 channel tests (plain ``pytest`` + ``numpy``) and the
S6 :mod:`~auditory_handshake.metrics.link_budget` dB-SPL contract: a rendered
source's RMS equals its ``level_db_spl``, and an incoherent sum of sources matches
``noise_floor_db`` (powers add, levels don't).
"""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.acoustics.noise import (
    SUPPORTED_KINDS,
    synthesize_noise,
    synthesize_noise_floor,
)
from auditory_handshake.config import NoiseSource
from auditory_handshake.metrics.link_budget import noise_floor_db

FS = 16000.0
DUR = 2.0
N_SAMPLES = round(DUR * FS)
REF_P = 20.0e-6


def _spl(x: np.ndarray) -> float:
    """RMS of a pressure waveform in dB SPL (re 20 µPa)."""
    return 20.0 * float(np.log10(np.sqrt(np.mean(np.square(x))) / REF_P))


def _power_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A single-bin power spectrum (magnitude squared) over rfft bins."""
    n = len(x)
    mag2 = np.square(np.abs(np.fft.rfft(x)))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    return freqs, mag2


def _band_power(freqs: np.ndarray, mag2: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    return float(np.sum(mag2[mask]))


def _source(kind: str, center_hz: float | None = None) -> NoiseSource:
    return NoiseSource(kind=kind, level_db_spl=60.0, distance_m=1.0, center_hz=center_hz)


KINDS = [
    pytest.param("broadband", None, id="broadband"),
    pytest.param("hvac_pink", None, id="hvac_pink"),
    pytest.param("speech", None, id="speech"),
    pytest.param("rotor", 600.0, id="rotor"),
]


# --- basic contract ---------------------------------------------------------
@pytest.mark.parametrize(("kind", "center_hz"), KINDS)
def test_shape_and_dtype(kind: str, center_hz: float | None) -> None:
    x = synthesize_noise(_source(kind, center_hz), DUR, FS, seed=42)
    assert x.shape == (N_SAMPLES,)
    assert x.dtype == np.float64
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize(("kind", "center_hz"), KINDS)
def test_spl_matches_level(kind: str, center_hz: float | None) -> None:
    source = _source(kind, center_hz)
    x = synthesize_noise(source, DUR, FS, seed=42)
    assert _spl(x) == pytest.approx(source.level_db_spl, abs=0.05)


@pytest.mark.parametrize(("kind", "center_hz"), KINDS)
def test_determinism_same_seed(kind: str, center_hz: float | None) -> None:
    source = _source(kind, center_hz)
    a = synthesize_noise(source, DUR, FS, seed=42)
    b = synthesize_noise(source, DUR, FS, seed=42)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(("kind", "center_hz"), KINDS)
def test_determinism_differs_by_seed(kind: str, center_hz: float | None) -> None:
    source = _source(kind, center_hz)
    a = synthesize_noise(source, DUR, FS, seed=42)
    b = synthesize_noise(source, DUR, FS, seed=7)
    assert not np.allclose(a, b)


def test_two_identical_sources_are_incoherent() -> None:
    # Two identical sources in one scenario must stay incoherent when summed, so
    # the floor is L + 3 dB (powers add), not L + 6 dB (amplitudes add). This
    # exercises the per-position ``salt`` that keeps identical sources independent.
    x = synthesize_noise_floor([_source("hvac_pink"), _source("hvac_pink")], DUR, FS, seed=42)
    expected = noise_floor_db([60.0, 60.0])  # == 60 + 10*log10(2) = 63.0103
    assert _spl(x) == pytest.approx(expected, abs=0.1)
    assert _spl(x) < 66.0  # guard: incoherent (L+3dB), not coherent (L+6dB)


# --- floor agreement with link budget (S6 contract) -------------------------
def test_noise_floor_matches_link_budget() -> None:
    sources = [
        NoiseSource("hvac_pink", 40.0, 1.0, None),
        NoiseSource("broadband", 55.0, 1.0, None),
    ]
    floor = synthesize_noise_floor(sources, DUR, FS, seed=42)
    expected = noise_floor_db([s.level_db_spl for s in sources])
    assert _spl(floor) == pytest.approx(expected, abs=0.1)


def test_noise_floor_empty() -> None:
    x = synthesize_noise_floor([], DUR, FS, seed=42)
    assert x.shape == (N_SAMPLES,)
    assert float(np.sum(x)) == 0.0


# --- spectral shape (physics) ----------------------------------------------
def test_rotor_has_dominant_tone_at_bpf() -> None:
    x = synthesize_noise(_source("rotor", 600.0), DUR, FS, seed=42)
    freqs, mag2 = _power_spectrum(x)
    peak_hz = float(freqs[int(np.argmax(mag2))])
    assert abs(peak_hz - 600.0) < 3.0
    # ...and the fundamental carries more power than its 5th-harmonic region.
    fundamental = _band_power(freqs, mag2, 560.0, 640.0)
    fifth_harm = _band_power(freqs, mag2, 2860.0, 2940.0)
    assert fundamental > fifth_harm


def test_rotor_rejects_out_of_band_bpf() -> None:
    with pytest.raises(ValueError):
        synthesize_noise(_source("rotor", 40000.0), DUR, FS, seed=42)


def test_hvac_pink_power_decays_with_frequency() -> None:
    x = synthesize_noise(_source("hvac_pink"), DUR, FS, seed=42)
    freqs, mag2 = _power_spectrum(x)

    def _per_bin(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(mag2[mask]) / max(1, int(np.count_nonzero(mask))))

    low = _per_bin(80.0, 160.0)
    high = _per_bin(1000.0, 2000.0)
    # Pink noise: per-bin PSD ∝ 1/f (each octave holds the *same* total power),
    # so the ~1 kHz bin is ≈ 1000/80 ≈ 12.5× weaker than the ~80 Hz bin.
    # Loose bounds absorb the statistical variance of a 2 s realisation.
    assert high < low / 5.0
    assert high > low * 0.02


def test_speech_energy_concentrated_in_voice_band() -> None:
    x = synthesize_noise(_source("speech"), DUR, FS, seed=42)
    freqs, mag2 = _power_spectrum(x)
    in_band = _band_power(freqs, mag2, 300.0, 4000.0)
    out_of_band = _band_power(freqs, mag2, 6000.0, 7000.0)
    assert in_band > out_of_band


def test_broadband_is_flat() -> None:
    x = synthesize_noise(_source("broadband"), DUR, FS, seed=42)
    freqs, mag2 = _power_spectrum(x)
    low = _band_power(freqs, mag2, 200.0, 400.0)
    high = _band_power(freqs, mag2, 3000.0, 3200.0)
    # Flat PSD: comparable power a wide frequency apart.
    assert low / high == pytest.approx(1.0, rel=0.5)


def test_supported_kinds_constant() -> None:
    # The public constant must match exactly the ``kind`` values shipped in configs.
    assert SUPPORTED_KINDS == frozenset({"broadband", "hvac_pink", "speech", "rotor"})
