"""Tests for :mod:`auditory_handshake.acoustics.motion` (S12.1, moving source).

These pin the S12.1 contract from PLAN §2.3: a source at closing speed ``v``
is heard at ``f_obs = f·c/(c − v)`` (``f_d = f·v/(c − v)``). So
:func:`moving_source` must shift a steady tone by exactly that factor,
up for approach / down for recession, stay phase-continuous across its
resampling blocks, remain deterministic and finite, honour a per-block speed
profile (the S12.2 entry point), and reject invalid geometry.
:func:`doppler_factor` is pinned to the analytic formula;
:func:`measure_doppler_hz` is pinned to the zero-crossing estimate it claims;
:func:`trajectory` to the piecewise-linear 1-D path.
"""

import numpy as np
import pytest

from auditory_handshake.acoustics import (
    SPEED_OF_SOUND_M_S,
    doppler_factor,
    measure_doppler_hz,
    moving_source,
    trajectory,
)

FS = 16000.0
C = SPEED_OF_SOUND_M_S


def _tone(freq_hz: float, seconds: float, fs: float = FS, amplitude: float = 1.0) -> np.ndarray:
    """A deterministic sine of ``freq_hz`` for ``seconds`` at ``fs``."""
    t = np.arange(int(round(seconds * fs))) / fs
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def test_doppler_factor_matches_analytic_formula() -> None:
    # f_obs = f·c/(c − v): the resampling factor is c/(c − v).
    assert doppler_factor(0.0) == pytest.approx(1.0)
    assert doppler_factor(5.0) == pytest.approx(C / (C - 5.0))
    assert doppler_factor(-5.0) == pytest.approx(C / (C + 5.0))
    # Approach raises pitch, recession lowers it.
    assert doppler_factor(5.0) > 1.0 > doppler_factor(-5.0)
    # Higher speed, higher factor; custom c threads through.
    assert doppler_factor(10.0) > doppler_factor(5.0)
    assert doppler_factor(5.0, speed_of_sound_m_s=300.0) == pytest.approx(300.0 / 295.0)


def test_doppler_factor_rejects_invalid_speed() -> None:
    for bad in (C, -C, C * 2, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="speed_ms"):
            doppler_factor(bad)
    with pytest.raises(ValueError, match="speed_of_sound_m_s"):
        doppler_factor(0.0, speed_of_sound_m_s=-C)


def test_stationary_source_is_the_identity() -> None:
    # v = 0 -> factor 1 -> the output must be the (resampled) source itself.
    sig = _tone(1000.0, 0.25)
    out = moving_source(sig, FS, 0.0, duration_seconds=0.25, block_duration_seconds=0.05)
    assert out.shape == sig.shape
    np.testing.assert_allclose(out, sig, rtol=1e-4, atol=1e-6)


def test_approach_shifts_pitch_up_by_formula() -> None:
    # PLAN §2.3: at 1 kHz and 5 m/s the shift is ~15 Hz upward.
    sig = _tone(1000.0, 1.0)
    out = moving_source(sig, FS, 5.0, duration_seconds=0.25, block_duration_seconds=0.05)
    expected = 1000.0 * C / (C - 5.0) - 1000.0  # ≈ +14.79 Hz
    assert measure_doppler_hz(out, FS, 1000.0) == pytest.approx(expected, abs=5.0)


def test_recession_shifts_pitch_down_by_formula() -> None:
    sig = _tone(1000.0, 1.0)
    out = moving_source(sig, FS, -5.0, duration_seconds=0.25, block_duration_seconds=0.05)
    expected = 1000.0 * C / (C + 5.0) - 1000.0  # ≈ −14.37 Hz
    assert measure_doppler_hz(out, FS, 1000.0) == pytest.approx(expected, abs=5.0)


def test_plan_reference_10k_hz_5_m_s() -> None:
    # PLAN §2.3 reference case: "at 10 kHz and 5 m/s → f_d ≈ 146 Hz".
    sig = _tone(10000.0, 1.0, fs=48000.0)
    out = moving_source(sig, 48000.0, 5.0, duration_seconds=0.25, block_duration_seconds=0.05)
    shift = measure_doppler_hz(out, 48000.0, 10000.0)
    assert shift == pytest.approx(10000.0 * 5.0 / (C - 5.0), abs=5.0)  # ≈ +147.9 Hz


def test_output_length_finite_and_deterministic() -> None:
    sig = _tone(1500.0, 2.0)
    a = moving_source(sig, FS, 5.0, duration_seconds=0.3, block_duration_seconds=0.1)
    b = moving_source(sig, FS, 5.0, duration_seconds=0.3, block_duration_seconds=0.1)
    assert a.size == int(round(0.3 * FS))
    assert np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)


def test_no_click_at_block_boundaries() -> None:
    # Walk positions accumulate across blocks, so the waveform must be
    # phase-continuous: no sample jump bigger than the tone's own slope.
    sig = _tone(1000.0, 1.0)
    out = moving_source(sig, FS, 5.0, duration_seconds=0.25, block_duration_seconds=0.01)
    max_slope = 2.0 * np.pi * (1000.0 * C / (C - 5.0)) / FS  # per-sample slope bound
    assert np.max(np.abs(np.diff(out))) < max_slope * 1.05


def test_variable_speed_profile_switches_shifts_per_block() -> None:
    # S12.2 entry point: one closing speed per output block, in time order.
    # (0.2 s halves keep the zero-crossing estimate well within tolerance.)
    sig = _tone(1000.0, 2.0)
    out = moving_source(sig, FS, [5.0, -5.0], duration_seconds=0.4, block_duration_seconds=0.2)
    half = out.size // 2
    first = measure_doppler_hz(out[:half], FS, 1000.0)
    second = measure_doppler_hz(out[half:], FS, 1000.0)
    assert first == pytest.approx(1000.0 * C / (C - 5.0) - 1000.0, abs=8.0)  # + approach
    assert second == pytest.approx(1000.0 * C / (C + 5.0) - 1000.0, abs=8.0)  # − recession


def test_variable_profile_rejects_wrong_length() -> None:
    sig = _tone(1000.0, 1.0)
    with pytest.raises(ValueError, match="per-block"):
        moving_source(sig, FS, [5.0, -5.0, 0.0], duration_seconds=0.2, block_duration_seconds=0.1)


def test_too_short_source_raises() -> None:
    # Approach consumes signal faster than real time: a 0.25 s window at 5 m/s
    # needs ~0.2537 s of source, so a 0.25 s source (4000 samples) is not
    # enough (needs ~4059), while 0.26 s is just enough.
    sig = _tone(1000.0, 0.25)
    with pytest.raises(ValueError, match="source samples"):
        moving_source(sig, FS, 5.0, duration_seconds=0.25, block_duration_seconds=0.05)
    # A receding source consumes less signal than real time: 0.25 s suffices.
    out = moving_source(sig, FS, -5.0, duration_seconds=0.25, block_duration_seconds=0.05)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    ("signal", "fs", "speed_ms", "duration_seconds", "block_duration_seconds"),
    [
        (_tone(1000.0, 1.0), FS, C, 0.1, 0.05),
        (_tone(1000.0, 1.0), FS, 5.0, 0.0, 0.05),
        (_tone(1000.0, 1.0), FS, 5.0, 0.1, 0.2),
        (_tone(1000.0, 1.0), -FS, 5.0, 0.1, 0.05),
        (_tone(1000.0, 1.0)[:, None], FS, 5.0, 0.1, 0.05),
    ],
)
def test_moving_source_rejects_invalid_geometry(
    signal: np.ndarray,
    fs: float,
    speed_ms: float,
    duration_seconds: float,
    block_duration_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        moving_source(
            signal,
            fs,
            speed_ms,
            duration_seconds=duration_seconds,
            block_duration_seconds=block_duration_seconds,
        )


def test_trajectory_approach_is_linear_and_reflected() -> None:
    pos, dist = trajectory(5.0, duration_seconds=10.0, initial_distance_m=100.0)
    t = np.arange(pos.size) / FS
    np.testing.assert_allclose(pos, 100.0 - 5.0 * t, rtol=1e-9)
    np.testing.assert_allclose(dist, np.abs(pos), rtol=1e-9)
    assert pos.size == int(round(10.0 * FS))
    # Last sample sits at t = (size−1)/FS, one sample short of 10 s.
    assert pos[-1] == pytest.approx(100.0 - 5.0 * (pos.size - 1) / FS)


def test_trajectory_recession_grows() -> None:
    pos, dist = trajectory(-2.0, duration_seconds=5.0, initial_distance_m=10.0)
    assert pos[-1] == pytest.approx(10.0 + 2.0 * (pos.size - 1) / FS)
    assert dist[-1] == pytest.approx(10.0 + 2.0 * (pos.size - 1) / FS)


def test_trajectory_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="initial_distance_m"):
        trajectory(5.0, duration_seconds=1.0, initial_distance_m=0.0)
    with pytest.raises(ValueError, match="speed_ms"):
        trajectory(C, duration_seconds=1.0, initial_distance_m=100.0)


def test_measure_doppler_hz_errors() -> None:
    short = _tone(1000.0, 0.001, fs=FS)
    with pytest.raises(ValueError, match="25 samples"):
        measure_doppler_hz(short, FS, 1000.0)
    with pytest.raises(ValueError, match="zero crossing"):
        measure_doppler_hz(np.ones(100), FS, 1000.0)
    with pytest.raises(ValueError, match="nominal_hz"):
        measure_doppler_hz(_tone(1000.0, 0.25), FS, 0.0)
