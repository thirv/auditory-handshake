"""Tests for the perceptual (mosqito) acoustic metrics (S17.1 annoyance model)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from auditory_handshake.acoustics.noise import synthesize_noise
from auditory_handshake.config import NoiseSource
from auditory_handshake.metrics import (
    PerceptualMetrics,
    noise_source_perceptual_metrics,
    perceptual_annoyance,
    perceptual_metrics,
)

FS = 16_000
DUR = 3.0  # >= 48_000 samples: stable ZWTV / DIN-45637 window at 16 kHz
MIN = 48_000


def test_exports_and_dataclass() -> None:
    assert isinstance(perceptual_metrics(np.zeros(4, dtype=np.float64), FS), PerceptualMetrics)
    m = PerceptualMetrics(loudness_sone=1.0, sharpness_acum=2.0, roughness_asper=3.0)
    assert m.as_dict() == {
        "loudness_sone": 1.0,
        "sharpness_acum": 2.0,
        "roughness_asper": 3.0,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.loudness_sone = 5.0  # frozen dataclass


def test_short_and_empty_signal_zero() -> None:
    zero = {"loudness_sone": 0.0, "sharpness_acum": 0.0, "roughness_asper": 0.0}
    for sig in (np.zeros(4, dtype=np.float64), np.zeros(int(1.0 * FS), dtype=np.float64)):
        assert perceptual_metrics(sig, FS).as_dict() == zero


def test_all_nan_signal_zero() -> None:
    sig = np.full(MIN + 1, np.nan, dtype=np.float64)
    zero = {"loudness_sone": 0.0, "sharpness_acum": 0.0, "roughness_asper": 0.0}
    assert perceptual_metrics(sig, FS).as_dict() == zero


def test_invalid_fs_raises() -> None:
    with pytest.raises(ValueError):
        perceptual_metrics(np.zeros(MIN + 1, dtype=np.float64), 0.0)
    with pytest.raises(ValueError):
        perceptual_metrics(np.zeros(MIN + 1, dtype=np.float64), float("nan"))


def test_real_metrics_finite_and_ordered() -> None:
    pytest.importorskip("mosqito")
    src = NoiseSource(kind="broadband", level_db_spl=70.0, distance_m=1.0)
    m = noise_source_perceptual_metrics(src, DUR, FS, seed=42)
    d = m.as_dict()
    assert all(np.isfinite(v) for v in d.values())
    assert d["loudness_sone"] > 0.0
    assert d["sharpness_acum"] > 0.0
    assert d["roughness_asper"] >= 0.0


def test_loudness_monotonic_in_level() -> None:
    pytest.importorskip("mosqito")
    lo = noise_source_perceptual_metrics(
        NoiseSource(kind="broadband", level_db_spl=50.0, distance_m=1.0), DUR, FS
    )
    hi = noise_source_perceptual_metrics(
        NoiseSource(kind="broadband", level_db_spl=80.0, distance_m=1.0), DUR, FS
    )
    assert hi.loudness_sone > lo.loudness_sone


def test_higher_frequency_is_sharper() -> None:
    pytest.importorskip("mosqito")
    t = np.arange(MIN) / FS

    def tone(hz: float, level_db: float = 60.0) -> np.ndarray:
        rms = 20e-6 * 10 ** (level_db / 20.0)
        amp = np.sqrt(2) * rms
        return amp * np.sin(2 * np.pi * hz * t)

    low = perceptual_metrics(tone(250.0), FS)
    high = perceptual_metrics(tone(4000.0), FS)
    assert high.sharpness_acum > low.sharpness_acum


def test_noise_source_convenience_matches_direct() -> None:
    pytest.importorskip("mosqito")
    src = NoiseSource(kind="hvac_pink", level_db_spl=65.0, distance_m=1.0)
    via_conv = noise_source_perceptual_metrics(src, DUR, FS, seed=7, salt=3)
    direct = perceptual_metrics(synthesize_noise(src, DUR, FS, seed=7, salt=3), FS)
    assert via_conv == direct


def test_annoyance_non_negative_below_reference() -> None:
    """A tolerable (below-reference) sound must score ``annoyance ≥ 0``.

    Regression test for the S17.1 clamp fix: ``_penalty_ramp`` originally
    returned the raw odd sigmoid (range ``(-1, 1)``), so a *below-reference*
    loudness drove the penalty — and hence ``annoyance = N_avg * penalty`` —
    **negative** (a tolerable 70 dB SPL broadband noise, N_avg ≈ 24.6 sone
    < 30 sone reference, scored a negative annoyance, i.e. "less annoying
    than silence"). The penalty is now clamped to ``[0, 1]`` ("at/below
    reference ⇒ 0"), so for every signal type the result must satisfy
    ``0 ≤ annoyance ≤ N_avg`` and be finite.
    """
    pytest.importorskip("mosqito")
    src = NoiseSource(kind="broadband", level_db_spl=70.0, distance_m=1.0)
    sig = synthesize_noise(src, DUR, FS, seed=7)
    for signal_type in ("noise", "sharp", "rough"):
        res = perceptual_annoyance(sig, FS, signal_type=signal_type)
        assert 0.0 <= res.annoyance <= res.N_avg
        assert math.isfinite(res.annoyance)


def test_modulation_factor_formula() -> None:
    """CSS/MFSK annoyance factor follows ``1 + (alpha*S)^2 + (beta*R)^2``.

    This is a pure-math check of :func:`_modulation_factor` (no ``mosqito``
    needed): the factor is always ``>= 1``, CSS weights sharpness and MFSK
    weights roughness, so each modulation type should dominate when that
    perceptual component is present.
    """
    from auditory_handshake.metrics.perceptual import _MODULATION_COEFFS, _modulation_factor

    for signal_type, (alpha, beta) in _MODULATION_COEFFS.items():
        for S_avg, R_avg in ((5.0, 3.0), (0.0, 0.0), (2.0, 0.0), (0.0, 4.0)):
            expected = 1.0 + (alpha * S_avg) ** 2 + (beta * R_avg) ** 2
            assert _modulation_factor(signal_type, S_avg, R_avg) == pytest.approx(expected)
            assert _modulation_factor(signal_type, S_avg, R_avg) >= 1.0

    # CSS sharpness-weighted (alpha=0.65), MFSK roughness-weighted (beta=0.60).
    assert _modulation_factor("css", 10.0, 0.0) > _modulation_factor("mfsk", 10.0, 0.0)
    assert _modulation_factor("mfsk", 0.0, 10.0) > _modulation_factor("css", 0.0, 10.0)


def test_css_mfsk_annoyance_matches_factor() -> None:
    """``annoyance = N_avg * factor`` for the modulated-carrier signal types."""
    pytest.importorskip("mosqito")
    from auditory_handshake.metrics.perceptual import _MODULATION_COEFFS

    sig = synthesize_noise(
        NoiseSource(kind="broadband", level_db_spl=75.0, distance_m=1.0), DUR, FS, seed=3
    )
    for signal_type in ("css", "mfsk"):
        res = perceptual_annoyance(sig, FS, signal_type=signal_type)
        alpha, beta = _MODULATION_COEFFS[signal_type]
        factor = 1.0 + (alpha * res.S_avg) ** 2 + (beta * res.R_avg) ** 2
        assert math.isfinite(res.annoyance)
        assert res.annoyance == pytest.approx(res.N_avg * factor)
        assert res.annoyance >= res.N_avg  # factor >= 1 for modulation types


def test_invalid_signal_type_or_noise_kind_raises() -> None:
    """Unknown ``signal_type``/``noise_kind`` values are rejected (no mosqito)."""
    short = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError):
        perceptual_annoyance(short, FS, signal_type="qam")
    with pytest.raises(ValueError):
        perceptual_annoyance(short, FS, signal_type="noise", noise_kind="white")
    # ``noise_kind`` is only validated for the ``"noise"`` signal type; css/mfsk
    # accept any noise_kind.
    perceptual_annoyance(short, FS, signal_type="css", noise_kind="white")
    perceptual_annoyance(short, FS, signal_type="mfsk", noise_kind="white")
