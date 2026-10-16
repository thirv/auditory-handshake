"""Tests for metrics.channel_metrics (S10.1 — RT60, τ_rms, B50/B90).

All expected values are derived analytically from the synthetic RIRs:

- Exponential ``r(t)=e^(−t/λ)`` → energy density ``e^(−2t/λ)``,
  Schroeder integral ``E(t)=E₀e^(−2t/λ)`` (exact →
  ``RT60 = ln(10⁶)·λ/2 ≈ 6.908·λ``); ``τ_mean = τ_rms = λ/2`` exactly;
  ``B50 = (λ/2)·ln 2``; ``B90 = (λ/2)·ln 10``.
- Linear amplitude ``r(t) = sqrt(2(D−t)/(3D))`` (uniform linear energy
  density, duration ``D = 3λ``) → ``τ_mean = D/3 = λ`` exactly,
  ``τ_rms = D/(3√2) ≈ 0.707·λ`` exactly (single-pulse: ``τ_rms² = σ²``),
  Schroeder integral ``E(t) ∝ (D−t)²`` is log-quadratic in the 5–30 ms window
  (→ ``RT60 ≈ 0.0590 s`` for ``λ = 10 ms``), ``B50 = D(1−1/√2) ≈ 0.2929·D``,
  ``B90 = D(1−√0.1) ≈ 0.6838·D`` (exact).
- Flat amplitude (uniform energy density) → ``E(t)`` linear in the tail,
  ``τ_rms = D/√12``, ``B50 = D/2``, ``B90 = 0.9D`` (exact).
- Single-sample impulse → ``RT60 = inf``, ``τ = 0``, ``B50 = B90 = 0``.
"""

import math

import numpy as np
import pytest

from auditory_handshake.metrics import channel_metrics as cm
from auditory_handshake.metrics.channel_metrics import (
    b50,
    b90,
    rt60_schroeder,
    tau_mean,
    tau_rms,
)

FS = 16000.0
LAMBDA = 0.010  # s — decay constant for the exponential/linear synthetic RIRs


def _exp_rir(lambda_s: float, tail_s: float = 0.300) -> np.ndarray:
    """r(t) = exp(−t/λ) sampled at FS (analytic Schroeder/τ/B references)."""
    t = np.arange(int(tail_s * FS)) / FS
    return np.exp(-t / lambda_s)


def _linear_rir(lambda_s: float, D: float) -> np.ndarray:
    """r(t) = sqrt(2(D−t)/(3D)) for t ∈ [0, D]: uniform linear energy density."""
    n = int(D * FS)
    t = np.arange(n) / FS
    e = 2.0 / (3.0 * D) * (D - t)  # e(D) = 0; integrates to 1 over [0, D]
    return np.sqrt(np.maximum(e, 0.0))


# --------------------------------------------------------------------------- #
# Module surface (package __all__ must match)
# --------------------------------------------------------------------------- #
def test_module_surface():
    for name in ["rt60_schroeder", "tau_mean", "tau_rms", "b50", "b90"]:
        assert hasattr(cm, name), f"channel_metrics missing {name}"
        assert name in cm.__all__


# --------------------------------------------------------------------------- #
# RT60 — Schroeder integral
# --------------------------------------------------------------------------- #
def test_rt60_exp_rir_recovers_lambdas():
    h = _exp_rir(LAMBDA)
    rt60 = rt60_schroeder(h, FS)
    exact = math.log(1e6) * LAMBDA / 2.0  # 6.9078 · λ
    assert rt60 == pytest.approx(exact, rel=2e-2)
    assert 0.05 < rt60 < 0.10


def test_rt60_linear_rir_window_fit():
    # Linear energy density e(t) ∝ (D−t) ⇒ Schroeder integral E(t) ∝ (D−t)²:
    # log-quadratic, so the least-squares slope over the 5–30 ms window is a
    # deterministic constant (not λ) — pin it against the analytic fit.
    D = 3.0 * LAMBDA
    h = _linear_rir(LAMBDA, D)
    rt60 = rt60_schroeder(h, FS)
    assert rt60 == pytest.approx(0.059020, rel=1e-3)


def test_rt60_anechoic_returns_inf():
    h = np.zeros(100)
    h[42] = 1.0  # single-sample "impulse"
    with pytest.warns(RuntimeWarning):
        assert rt60_schroeder(h, FS) == float("inf")


def test_rt60_single_sample_returns_inf():
    with pytest.warns(RuntimeWarning):
        assert rt60_schroeder(np.array([1.0]), FS) == float("inf")


def test_rt60_window_clamped_and_window_validated():
    h = _exp_rir(LAMBDA)
    # t1 beyond the RIR tail: clamped, still finite.
    assert np.isfinite(rt60_schroeder(h, FS, window=(0.005, 5.0)))
    # Reversed window → ValueError.
    with pytest.raises(ValueError, match="t0"):
        rt60_schroeder(h, FS, window=(0.030, 0.005))
    # Window smaller than 1 sample at this fs → ValueError.
    with pytest.raises(ValueError, match="too small|2 samples"):
        rt60_schroeder(h, FS, window=(0.0, 0.4 / FS))  # sub-sample


def test_rt60_rejects_bad_inputs():
    with pytest.raises(ValueError):
        rt60_schroeder(np.zeros((3, 3)), FS)  # 2-D
    with pytest.raises(ValueError):
        rt60_schroeder(np.array([]), FS)  # empty
    with pytest.raises(ValueError):
        rt60_schroeder(_exp_rir(LAMBDA), -1.0)  # fs <= 0
    with pytest.raises(ValueError):
        rt60_schroeder(np.array([1.0, np.nan]), FS)  # non-finite


# --------------------------------------------------------------------------- #
# Delay spread (τ_mean / τ_rms)
# --------------------------------------------------------------------------- #
def test_tau_exp_rir():
    h = _exp_rir(LAMBDA)
    assert tau_mean(h, FS) == pytest.approx(LAMBDA / 2.0, rel=2e-2)
    assert tau_rms(h, FS) == pytest.approx(LAMBDA / 2.0, rel=2e-2)


def test_tau_linear_rir_exact():
    D = 3.0 * LAMBDA
    h = _linear_rir(LAMBDA, D)
    assert tau_mean(h, FS) == pytest.approx(D / 3.0, rel=2e-2)
    assert tau_rms(h, FS) == pytest.approx(D / (3.0 * math.sqrt(2.0)), rel=2e-2)


def test_tau_anechoic_is_zero():
    h = np.zeros(100)
    h[42] = 1.0
    assert tau_mean(h, FS) == 42.0 / FS
    assert tau_rms(h, FS) == pytest.approx(0.0, abs=1e-12)


def test_tau_single_sample_is_zero():
    assert tau_mean(np.array([1.0]), FS) == 0.0
    assert tau_rms(np.array([1.0]), FS) == 0.0


def test_tau_two_taps_energy_split():
    # Two delta taps: 25% energy at 0 s, 75% at 10 ms ⇒ exact moments.
    h = np.zeros(int(0.050 * FS))
    h[0] = 1.0
    h[int(0.010 * FS)] = np.sqrt(3.0)
    assert tau_mean(h, FS) == pytest.approx(0.0075, rel=1e-2)
    assert tau_rms(h, FS) == pytest.approx(math.sqrt(0.0075 * 0.0025), rel=1e-2)


# --------------------------------------------------------------------------- #
# B50 / B90 — early/late energy fraction times
# --------------------------------------------------------------------------- #
def test_b50_b90_exp_rir():
    h = _exp_rir(LAMBDA)
    assert b50(h, FS) == pytest.approx(LAMBDA * 0.5 * math.log(2.0), rel=2e-2)
    assert b90(h, FS) == pytest.approx(LAMBDA * 0.5 * math.log(10.0), rel=2e-2)


def test_b50_b90_linear_rir_exact():
    D = 3.0 * LAMBDA
    h = _linear_rir(LAMBDA, D)
    assert b50(h, FS) == pytest.approx(D * (1.0 - 1.0 / math.sqrt(2.0)), rel=2e-2)
    assert b90(h, FS) == pytest.approx(D * (1.0 - math.sqrt(0.1)), rel=2e-2)


def test_b50_b90_flat_rir():
    D = 0.050
    h = np.ones(int(D * FS))
    assert b50(h, FS) == pytest.approx(D / 2.0, rel=5e-2)
    assert b90(h, FS) == pytest.approx(0.9 * D, rel=5e-2)


def test_b50_b90_anechoic_is_zero():
    h = np.zeros(100)
    h[42] = 1.0
    assert b50(h, FS) == 42.0 / FS  # fraction reached at the impulse sample
    assert b90(h, FS) == 42.0 / FS


# --------------------------------------------------------------------------- #
# Real RIR (pyroomacoustics) — sanity
# --------------------------------------------------------------------------- #
def test_rt60_real_rir_sanity():
    from auditory_handshake.acoustics.channel import _tail_length_seconds, compute_rir
    from auditory_handshake.config import load_room_preset

    preset = load_room_preset("conference_room")
    rir = compute_rir(preset, rt60_seconds=1.0, fs=FS)
    assert rir.shape == (int(_tail_length_seconds(1.0) * FS),)
    rt60 = rt60_schroeder(rir, FS)
    # 5–30 ms window fit on a 1 s RT60 room: the slope is noisy (the Schroeder
    # integral is dominated by the direct-sound onset over a 25 ms window), so
    # we only require a plausible positive, finite value.
    assert 0.05 < rt60 < 5.0
    assert 0.0 < tau_mean(rir, FS) < 0.1
    assert 0.0 < tau_rms(rir, FS) < 0.1
    assert 0.0 < b50(rir, FS) < 0.05
    assert b50(rir, FS) <= b90(rir, FS)


# --------------------------------------------------------------------------- #
# Real RIR (pyroomacoustics) — RT60 sensitivity (DRR regression guard)
# --------------------------------------------------------------------------- #
def test_metrics_track_rt60_target():
    """Regression: decay-tail metrics must grow with the target RT60.

    ``_SRC_MIC_OFFSET_M`` sets the source/mic separation, which fixes the
    direct-to-reverb ratio. A centimetre-scale gap let the direct path swamp
    every reflection, locking ``b90`` / ``tau_rms`` / ``tau_mean`` / the
    Schroeder RT60 to the direct spike so they were *constant* across RT60
    targets. A realistic separation restores a plausible DRR, so these metrics
    are strictly larger for a more reverberant target on the same preset.
    ``b50`` is intentionally excluded: 50% of the energy legitimately sits in
    the direct path + earliest reflections, so it is near-constant by design.
    """
    from auditory_handshake.acoustics.channel import analyze_channel
    from auditory_handshake.config import load_room_preset

    preset = load_room_preset("conference_room")
    low = analyze_channel(preset, 0.2, fs=FS)
    high = analyze_channel(preset, 2.0, fs=FS)

    for name in ("rt60_schroeder", "b90", "tau_rms", "tau_mean"):
        assert getattr(high, name) > getattr(low, name), (
            f"{name} must increase with the target RT60; "
            f"got low={getattr(low, name)}, high={getattr(high, name)}"
        )


# --------------------------------------------------------------------------- #
# Real RIR (pyroomacoustics) — DRR band (source/mic separation sanity)
# --------------------------------------------------------------------------- #
def test_drr_plausible_band():
    """Regression: the DRR must land in a plausible band for real presets.

    DRR = 20 log10(E_early / E_late) — the ratio of the early energy to the
    remaining reverberant-tail energy.  The early window is ``30 ms``: ~10x
    the 0.3 m direct-path arrival delay (~1 ms) and consistent with the
    ``b90 < 50 ms`` decay bound in :func:`test_rt60_real_rir_sanity`.  A 1 ms
    window is too narrow — these are front-loaded reflection RIRs (the direct
    path is the head of a decaying echo train, not a free-field delta at index
    0), so 1 ms captures almost no energy and DRR goes deeply negative even
    though the channel is healthy.

    A healthy, usable channel has its direct + early-reflection energy
    dominate the late reverb by a wide margin, yet still leave a measurable
    reverb tail (so the decay-tail metrics are meaningful): DRR in roughly
    ``5–65 dB`` (early energy 3x–2000x the late energy).  Below that the RIR
    is reverb-dominated (no usable direct/early component); above it the RIR
    is degenerate (near-impulse).  DRR is geometry-dependent (cabin ~10 dB,
    small hall ~54 dB at RT60 = 0.3 s), so the band is deliberately broad but
    still excludes grossly broken RIRs (all-reverb, pure-impulse, NaN/zero).
    """
    from auditory_handshake.acoustics.channel import compute_rir
    from auditory_handshake.config import list_room_presets, load_room_preset

    rt60 = 0.3
    for name in list_room_presets():
        preset = load_room_preset(name)
        rir = compute_rir(preset, rt60_seconds=rt60, fs=FS)
        e = np.square(np.abs(rir), dtype=np.float64)
        e = e / e.sum()
        n_early = max(1, int(round(30e-3 * FS)))  # 30 ms early window (direct + early reflections)
        early = float(e[:n_early].sum())
        tail = float(e[n_early:].sum())
        assert tail > 0.0, f"{name}: no reverberant tail energy for DRR"
        drr_db = 20.0 * math.log10(early / tail)
        assert (
            5.0 <= drr_db <= 65.0
        ), f"{name}: DRR(30 ms)={drr_db:.2f} dB outside plausible 5–65 dB band"


def test_rir_direct_onset_at_zero() -> None:
    """The direct-sound onset must be anchored at index 0 (no leading gap).

    ``compute_rir`` re-anchors the first significant sample to index 0, so the
    channel never starts with a silent head (which would inflate perceived
    latency and desynchronise the direct path).  The RIR head must therefore
    already carry a meaningful share of its peak — a late-echo-only RIR would
    have ~0 energy here.
    """
    from auditory_handshake.acoustics.channel import compute_rir
    from auditory_handshake.config import list_room_presets, load_room_preset

    for name in list_room_presets():
        preset = load_room_preset(name)
        rir = compute_rir(preset, rt60_seconds=0.3, fs=FS)
        peak = float(np.max(np.abs(rir)))
        assert peak > 0.0, f"{name}: degenerate RIR (zero peak)"
        head = float(np.max(np.abs(rir[:8])))
        assert head >= 1e-3 * peak, (
            f"{name}: direct-sound onset not at index 0 " f"(head={head:.3e}, peak={peak:.3e})"
        )


def test_src_mic_offset_changes_rir() -> None:
    """The exposed ``src_mic_offset_m`` must actually change the RIR.

    Guards against the offset being silently ignored (hardcoded back to a fixed
    value).  A larger source-mic separation changes the direct-path delay and
    the reflection geometry, so the generated RIR must differ.
    """
    from auditory_handshake.acoustics.channel import compute_rir
    from auditory_handshake.config import load_room_preset

    preset = load_room_preset("cabin")
    a = compute_rir(preset, rt60_seconds=0.3, fs=FS, src_mic_offset_m=0.3)
    b = compute_rir(preset, rt60_seconds=0.3, fs=FS, src_mic_offset_m=0.5)
    assert not np.allclose(a, b), "src_mic_offset_m had no effect on the RIR"
    assert (
        float(np.max(np.abs(a - b))) > 1e-3
    ), "src_mic_offset_m effect on the RIR is negligible (parameter likely ignored)"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
