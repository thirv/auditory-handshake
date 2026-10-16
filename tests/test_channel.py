"""Tests for :func:`auditory_handshake.acoustics.channel.analyze_channel` (S10.2).

These pin the *binding* contract: :class:`ChannelAnalysis` carries the room
preset, the deterministic RIR, and the S10.1 channel metrics (Schroeder RT60,
``tau_mean``, ``tau_rms``, ``B50``/``B90``) together, is serialisable via
``to_dict`` (JSON/YAML safe, the shape ``configs/phy_constraints.yaml``
consumes), and reproduces the exact values the S10.1 metric functions return on
the same RIR. The metric math itself is covered by ``test_channel_metrics.py``.
"""

import json

import numpy as np
import pytest

from auditory_handshake.acoustics import (
    ChannelAnalysis,
    analyze_channel,
    compute_rir,
    compute_rir_sweep,
    echo_tail_guard_s,
    sweep_rir_matrix,
)
from auditory_handshake.acoustics.channel import _tail_length_seconds
from auditory_handshake.config import RoomPreset, list_room_presets, load_room_preset
from auditory_handshake.metrics.channel_metrics import b50, b90, rt60_schroeder, tau_mean, tau_rms

FS = 16000.0


def _flat(alpha: float, dims: tuple[float, float, float] = (5.0, 4.0, 3.0)) -> RoomPreset:
    """A frequency-flat room with uniform absorption on every face."""
    return RoomPreset(
        name="test",
        dimensions_m=dims,
        frequencies_hz=(1000.0,),
        absorption={"x": (alpha,), "y": (alpha,), "z": (alpha,)},
    )


def test_analysis_carries_preset_rir_and_metrics() -> None:
    preset = _flat(0.1)
    a = analyze_channel(preset, 0.8, fs=FS)

    assert isinstance(a, ChannelAnalysis)
    assert a.preset is preset
    assert a.rt60_target_seconds == pytest.approx(0.8)
    assert a.fs == pytest.approx(FS)

    # RIR: same well-formed, unit-peak, deterministic response as compute_rir.
    np.testing.assert_array_equal(a.rir, compute_rir(preset, 0.8, fs=FS))
    assert a.rir.ndim == 1
    assert a.rir.size == pytest.approx(_tail_length_seconds(0.8) * FS)
    assert np.all(np.isfinite(a.rir))
    assert float(np.max(np.abs(a.rir))) == pytest.approx(1.0)
    assert a.tail_seconds == pytest.approx(_tail_length_seconds(0.8))


def test_metrics_match_s10_1_functions_on_same_rir() -> None:
    # The analysis must be exactly what the S10.1 functions return on the same
    # RIR — no re-derivation or rounding divergence.
    preset = _flat(0.1)
    a = analyze_channel(preset, 0.8, fs=FS)

    assert a.rt60_schroeder == pytest.approx(rt60_schroeder(a.rir, FS))
    assert a.tau_mean == pytest.approx(tau_mean(a.rir, FS))
    assert a.tau_rms == pytest.approx(tau_rms(a.rir, FS))
    assert a.b50 == pytest.approx(b50(a.rir, FS))
    assert a.b90 == pytest.approx(b90(a.rir, FS))


def test_metrics_are_finite_and_ordered() -> None:
    a = analyze_channel(_flat(0.1), 0.8, fs=FS)

    for value in (a.rt60_schroeder, a.tau_mean, a.tau_rms, a.b50, a.b90):
        assert np.isfinite(value), value
        assert value >= 0.0

    # For a real decaying RIR: the 50% energy time must precede the 90% energy
    # time. (No tau_rms < tau_mean assertion: a sparse delay distribution with a
    # few strong late arrivals is well above its energy-weighted mean, so
    # tau_rms can legitimately exceed tau_mean.)
    assert a.b50 < a.b90
    # Sanity: the Schroeder estimate is a *same-order* proxy for the target, not
    # a tight read. It is a local 5-30 ms window fit (see
    # channel_metrics.rt60_schroeder) whose value is modulated by the
    # early-reflection comb and the source/mic geometry. With the corrected,
    # realistic DRR (see channel._SRC_MIC_OFFSET_M) it lands within an order of
    # magnitude of the 0.8 s target while remaining well above a degenerate 0.
    # (The constant-metrics defect this guards against is covered by
    # test_metrics_track_rt60_target in test_channel_metrics.py.)
    assert 0.8 / 3.0 < a.rt60_schroeder < 0.8 * 3.0


def test_rt60_zero_is_still_a_valid_analysis() -> None:
    # A near-direct impulse is a legitimate channel: metrics must still be
    # finite (the Schroeder slope still fits a decay on the residual tail).
    a = analyze_channel(_flat(0.1), 0.0, fs=FS)
    assert np.isfinite(a.rt60_schroeder)
    assert a.rt60_schroeder >= 0.0
    assert np.all(np.isfinite(a.rir))


@pytest.mark.parametrize("name", list_room_presets())
def test_analysis_for_every_preset_is_well_formed(name: str) -> None:
    preset = load_room_preset(name)
    a = analyze_channel(preset, 0.5, fs=FS)

    assert a.preset.name == name
    assert a.rir.ndim == 1
    assert a.rir.size > 0
    assert np.all(np.isfinite(a.rir))
    assert a.rt60_schroeder >= 0.0
    assert a.tau_mean >= 0.0
    assert a.tau_rms >= 0.0


def test_analysis_is_deterministic() -> None:
    a = analyze_channel(_flat(0.1), 0.8, fs=FS)
    b = analyze_channel(_flat(0.1), 0.8, fs=FS)
    np.testing.assert_array_equal(a.rir, b.rir)
    assert a.to_dict() == b.to_dict()


def test_to_dict_is_json_serializable_and_complete() -> None:
    preset = _flat(0.1)
    a = analyze_channel(preset, 0.8, fs=FS)
    d = a.to_dict()

    # JSON round-trip: no ndarrays, no non-finite floats (nulls only).
    assert json.loads(json.dumps(d)) == d

    assert d["rt60_target_seconds"] == pytest.approx(0.8)
    assert d["fs"] == pytest.approx(FS)
    assert d["rir_samples"] == a.rir.size
    assert d["preset"]["name"] == preset.name
    for key in ("rt60_schroeder", "tau_mean", "tau_rms", "b50", "b90"):
        value = d[key]
        assert value is None or isinstance(value, float)


def test_sample_rate_is_threaded_through() -> None:
    lo = analyze_channel(_flat(0.1), 0.8, fs=16000.0)
    hi = analyze_channel(_flat(0.1), 0.8, fs=48000.0)
    assert lo.fs == pytest.approx(16000.0)
    assert hi.fs == pytest.approx(48000.0)
    # Same physical tail (0.25 s floor / 3x RT60) at different rates -> proportional lengths.
    assert hi.rir.size / lo.rir.size == pytest.approx(3.0, rel=0.05)


# ---------------------------------------------------------------------------
# RT60 sweep: sweep_rir_matrix + compute_rir_sweep
# ---------------------------------------------------------------------------


def test_sweep_matrix_rows_match_individual_compute_rir() -> None:
    preset = _flat(0.1)
    targets = [0.2, 0.8, 1.5]
    matrix = sweep_rir_matrix(preset, targets, fs=FS)

    assert matrix.ndim == 2
    assert matrix.shape[0] == len(targets)
    assert matrix.dtype == np.float32
    assert np.all(np.isfinite(matrix))

    for i, v in enumerate(targets):
        expected = np.asarray(compute_rir(preset, v, fs=FS), dtype=np.float32)
        # Row is exactly the per-target RIR, zero-padded to the longest tail width.
        np.testing.assert_array_equal(matrix[i, : expected.size], expected)


def test_sweep_matrix_width_is_longest_tail() -> None:
    preset = _flat(0.1)
    targets = [0.2, 0.8, 1.5]
    matrix = sweep_rir_matrix(preset, targets, fs=FS)

    expected_width = int(round(_tail_length_seconds(max(targets)) * FS))
    assert matrix.shape[1] == expected_width

    # The longest-tail row is exactly full; shorter rows are left-padded (zeros)
    # past their own natural length.
    longest_row = matrix[targets.index(1.5)]
    assert longest_row.size == expected_width
    assert np.max(np.abs(longest_row)) == pytest.approx(1.0)

    short_row = matrix[targets.index(0.2)]
    natural_len = int(round(_tail_length_seconds(0.2) * FS))
    assert natural_len < expected_width
    assert np.all(short_row[natural_len:] == 0.0)


def test_sweep_matrix_single_value() -> None:
    preset = _flat(0.1)
    matrix = sweep_rir_matrix(preset, [0.8], fs=FS)

    assert matrix.ndim == 2
    assert matrix.shape[0] == 1
    expected = np.asarray(compute_rir(preset, 0.8, fs=FS), dtype=np.float32)
    np.testing.assert_array_equal(matrix[0, : expected.size], expected)
    assert np.max(np.abs(matrix[0])) == pytest.approx(1.0)


def test_sweep_matrix_accepts_non_list_sequence() -> None:
    preset = _flat(0.1)
    matrix = sweep_rir_matrix(preset, (0.2, 0.8), fs=FS)
    assert matrix.shape[0] == 2


def test_sweep_matrix_rejects_empty_sequence() -> None:
    preset = _flat(0.1)
    with pytest.raises(ValueError, match="non-empty"):
        sweep_rir_matrix(preset, [], fs=FS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_sweep_matrix_rejects_non_finite(bad: float) -> None:
    preset = _flat(0.1)
    with pytest.raises(ValueError, match="finite"):
        sweep_rir_matrix(preset, [0.2, bad], fs=FS)


def test_sweep_matrix_rejects_negative() -> None:
    preset = _flat(0.1)
    with pytest.raises(ValueError, match="non-negative"):
        sweep_rir_matrix(preset, [0.2, -0.1], fs=FS)


def test_sweep_matrix_threads_fs_and_offset() -> None:
    preset = _flat(0.1)
    targets = [0.2, 0.8]

    hi = sweep_rir_matrix(preset, targets, fs=48000.0)
    lo = sweep_rir_matrix(preset, targets, fs=16000.0)
    # Same physical tail at different rates -> proportional widths.
    assert hi.shape[1] / lo.shape[1] == pytest.approx(3.0, rel=0.05)

    # Explicit separation is threaded through to every row.
    custom = sweep_rir_matrix(preset, targets, fs=FS, src_mic_offset_m=0.05)
    for i, v in enumerate(targets):
        expected = np.asarray(
            compute_rir(preset, v, fs=FS, src_mic_offset_m=0.05), dtype=np.float32
        )
        np.testing.assert_array_equal(custom[i, : expected.size], expected)


def test_compute_rir_sweep_matches_individual_analyze_channel() -> None:
    preset = _flat(0.1)
    targets = [0.2, 0.8, 1.5]
    result = compute_rir_sweep(preset, targets, fs=FS)

    assert len(result) == len(targets)
    for analysis, v in zip(result, targets, strict=True):
        expected = analyze_channel(preset, v, fs=FS)
        assert isinstance(analysis, ChannelAnalysis)
        assert analysis.rt60_target_seconds == pytest.approx(v)
        np.testing.assert_array_equal(analysis.rir, expected.rir)
        assert analysis.to_dict() == expected.to_dict()


def test_compute_rir_sweep_single_value() -> None:
    preset = _flat(0.1)
    result = compute_rir_sweep(preset, [0.8], fs=FS)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].rir, compute_rir(preset, 0.8, fs=FS))


def test_compute_rir_sweep_rejects_empty_sequence() -> None:
    preset = _flat(0.1)
    with pytest.raises(ValueError, match="non-empty"):
        compute_rir_sweep(preset, [], fs=FS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1])
def test_compute_rir_sweep_rejects_invalid_values(bad: float) -> None:
    preset = _flat(0.1)
    with pytest.raises(ValueError):
        compute_rir_sweep(preset, [0.2, bad], fs=FS)


# ---------------------------------------------------------------------------
# echo_tail_guard_s
# ---------------------------------------------------------------------------
def _single_pole(n: int, fs: float, tau_s: float) -> np.ndarray:
    """A single-pole exponential RIR: h[n] = r^n, r = exp(-1 / (tau * fs))."""
    idx = np.arange(n)
    r = float(np.exp(-1.0 / (tau_s * fs)))
    h = r**idx
    return h / float(np.max(np.abs(h)))


def test_echo_tail_guard_matches_analytic_single_pole() -> None:
    fs = 16000.0
    tau = 0.3
    h = _single_pole(int(6 * tau * fs) + 64, fs, tau)
    measured = echo_tail_guard_s(h, fs, threshold_db=-30.0)
    expected = tau * np.log(10.0) * 30.0 / 20.0  # tau * ln(10) * |db| / 20
    assert measured == pytest.approx(expected, rel=0.02)


def test_echo_tail_guard_longer_decay_is_longer() -> None:
    fs = 16000.0
    fast = _single_pole(int(8 * 0.15 * fs) + 64, fs, 0.15)
    slow = _single_pole(int(8 * 0.45 * fs) + 64, fs, 0.45)
    assert echo_tail_guard_s(fast, fs) < echo_tail_guard_s(slow, fs)


def test_echo_tail_guard_threshold_monotonic() -> None:
    fs = 16000.0
    h = _single_pole(int(6 * 0.3 * fs) + 64, fs, 0.3)
    # Deeper decay -> longer measured tail.
    assert echo_tail_guard_s(h, fs, threshold_db=-20.0) < echo_tail_guard_s(
        h, fs, threshold_db=-60.0
    )


def test_echo_tail_guard_capped_at_rir_length_warns() -> None:
    fs = 16000.0
    n = 800
    # A short flat burst: the last sample still carries more than 0.1% of the
    # total energy, so the tail cannot decay to -30 dB below peak within it.
    h = np.ones(n, dtype=np.float64)
    with pytest.warns(RuntimeWarning, match="does not decay"):
        result = echo_tail_guard_s(h, fs, threshold_db=-30.0)
    assert result == pytest.approx(n / fs)


def test_echo_tail_guard_edge_cases() -> None:
    fs = 16000.0
    assert echo_tail_guard_s(np.ones(8, dtype=np.float64), fs, threshold_db=0.0) == 0.0
    assert echo_tail_guard_s(np.ones(8, dtype=np.float64), fs, threshold_db=5.0) == 0.0
    assert echo_tail_guard_s(np.zeros(16, dtype=np.float64), fs) == 0.0
    assert echo_tail_guard_s(np.array([], dtype=np.float64), fs) == 0.0


@pytest.mark.parametrize("fs", [-1.0, 0.0])
def test_echo_tail_guard_rejects_bad_fs(fs: float) -> None:
    with pytest.raises(ValueError, match="fs"):
        echo_tail_guard_s(np.ones(16, dtype=np.float64), fs)


def test_echo_tail_guard_rejects_non_finite_threshold() -> None:
    with pytest.raises(ValueError, match="threshold_db"):
        echo_tail_guard_s(np.ones(16, dtype=np.float64), 16000.0, threshold_db=float("nan"))
