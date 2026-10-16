"""Tests for :mod:`auditory_handshake.metrics.ttfb`.

Covers: field fidelity against ``measure_detection``, the seconds→milliseconds
conversion, NaN handling (no detected trials), and custom target thresholds.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from auditory_handshake.metrics import TTCH_TTFB_TARGET_MS, TTFBReport, compute_ttfb
from auditory_handshake.phy import (
    PREAMBLE_WAVES,
    PreambleConfig,
    PreambleDetectionStats,
    measure_detection,
)

CFG = PreambleConfig()  # defaults: zadoff-chu, 61 samples @ 48 kHz
N_TRIALS = 20
SEED = 1
SNR_DB = 20.0
BUFFER_S = 0.125
MIN_OFFSET_FRAC = 0.2
MAX_OFFSET_FRAC = 0.7


def test_report_field_fidelity() -> None:
    """TTFBReport mirrors the underlying PreambleDetectionStats exactly."""
    stats = measure_detection(
        CFG,
        snr_db=SNR_DB,
        n_trials=N_TRIALS,
        buffer_s=BUFFER_S,
        min_offset_frac=MIN_OFFSET_FRAC,
        max_offset_frac=MAX_OFFSET_FRAC,
        seed=SEED,
    )
    report = compute_ttfb(
        CFG,
        snr_db=SNR_DB,
        n_trials=N_TRIALS,
        buffer_s=BUFFER_S,
        min_offset_frac=MIN_OFFSET_FRAC,
        max_offset_frac=MAX_OFFSET_FRAC,
        seed=SEED,
    )

    assert report.n_trials == stats.n_trials
    assert report.n_signal_trials == stats.n_signal_trials
    assert report.n_noise_trials == stats.n_noise_trials
    assert report.n_detected == stats.n_detected
    assert report.n_false_alarms == stats.n_false_alarms
    assert report.detection_probability == stats.detection_probability
    assert report.false_alarm_rate == stats.false_alarm_rate
    assert report.ttfb_p50_ms == pytest.approx(stats.latency_p50_s * 1000.0)
    assert report.ttfb_p99_ms == pytest.approx(stats.latency_p99_s * 1000.0)
    assert report.snr_db == stats.snr_db
    assert report.seed == stats.seed
    assert report.wave == CFG.wave
    assert report.target_ms == pytest.approx(TTCH_TTFB_TARGET_MS)


def test_milliseconds_conversion() -> None:
    """ms values are exactly 1000× the underlying seconds from the stats."""
    stats = measure_detection(CFG, snr_db=SNR_DB, n_trials=N_TRIALS, seed=SEED)
    report = compute_ttfb(CFG, snr_db=SNR_DB, n_trials=N_TRIALS, seed=SEED)
    assert report.ttfb_p50_ms == pytest.approx(stats.latency_p50_s * 1000.0)
    assert report.ttfb_p99_ms == pytest.approx(stats.latency_p99_s * 1000.0)

    # Latency floor is the preamble tail: (length - 1) / fs ~= 1.25 ms.
    tail_ms = (CFG.length - 1) / CFG.fs * 1000.0
    assert tail_ms <= report.ttfb_p50_ms <= tail_ms + 1e-6
    assert tail_ms <= report.ttfb_p99_ms <= tail_ms + 1e-6


def test_nan_handling_no_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no trial ever detects, latencies are NaN and achieved is False.

    The detector is energy-normalised (``|x ⊛ p| / ||p||²``), so an
    embedded preamble scores ``1.0 + N(0, 1/SNR)`` regardless of its
    amplitude — no real-signal configuration can guarantee a
    zero-detection run (any sub-1.0 threshold fires on about half the
    trials).  The zero-detection branch is therefore exercised through
    the measurement contract: pin ``measure_detection`` to a
    zero-detection stats object and assert the metrics layer renders it
    correctly (NaN latencies, ``achieved is False``).
    """
    cfg = PreambleConfig()
    stats = PreambleDetectionStats(
        n_trials=N_TRIALS,
        n_signal_trials=N_TRIALS // 2,
        n_noise_trials=N_TRIALS // 2,
        n_detected=0,
        n_false_alarms=0,
        detection_probability=0.0,
        false_alarm_rate=0.0,
        latency_p50_s=float("nan"),
        latency_p99_s=float("nan"),
        snr_db=-20.0,
        seed=SEED,
        config=cfg,
    )
    monkeypatch.setattr(
        "auditory_handshake.metrics.ttfb.measure_detection",
        lambda *_args, **_kwargs: stats,
    )
    report = compute_ttfb(cfg, snr_db=-20.0, n_trials=N_TRIALS, seed=SEED)
    assert report.n_detected == 0
    assert report.detection_probability == 0.0
    assert math.isnan(report.ttfb_p50_ms)
    assert math.isnan(report.ttfb_p99_ms)
    assert report.achieved is False


def test_positive_path_all_trials_detected() -> None:
    """Positive path: at high SNR every trial locks at the latency floor.

    The energy-normalised peak is ``1.0 + N(0, 1/SNR)``; at 100 dB the
    deviation is ~1e-5, far below the 0.7 decision threshold, so every
    signal trial is detected at its true offset and no noise-only trial
    fires.  This pins the happy path deterministically: p50 == p99 at the
    preamble-tail floor and the M4 target is met.
    """
    report = compute_ttfb(CFG, snr_db=100.0, n_trials=N_TRIALS, seed=SEED)
    assert report.n_detected == N_TRIALS // 2
    assert report.n_signal_trials == N_TRIALS // 2
    assert report.detection_probability == 1.0
    assert report.n_false_alarms == 0
    assert report.false_alarm_rate == 0.0
    tail_ms = (CFG.length - 1) / CFG.fs * 1000.0
    assert report.ttfb_p50_ms == pytest.approx(tail_ms)
    assert report.ttfb_p99_ms == pytest.approx(tail_ms)
    assert report.ttfb_p99_ms < report.target_ms
    assert report.achieved is True


def test_target_and_achieved_flag() -> None:
    """achieved reflects p99 vs target; tight target fails, loose target passes."""
    report = compute_ttfb(CFG, snr_db=SNR_DB, n_trials=N_TRIALS, seed=SEED)
    assert report.target_ms == pytest.approx(TTCH_TTFB_TARGET_MS)
    assert report.achieved is True

    tight = compute_ttfb(
        CFG,
        snr_db=SNR_DB,
        n_trials=N_TRIALS,
        seed=SEED,
        target_ms=report.ttfb_p99_ms * 0.01,
    )
    assert tight.target_ms == pytest.approx(report.ttfb_p99_ms * 0.01)
    assert tight.achieved is False


def test_all_waves_pass_m4() -> None:
    """Every preamble wave meets the M4 p99 TTFB exit criterion."""
    for wave in PREAMBLE_WAVES:
        cfg = PreambleConfig(wave=wave)
        report = compute_ttfb(cfg, snr_db=SNR_DB, n_trials=N_TRIALS, seed=SEED)
        # Per-trial detection at the 20 dB reference point is ≈certain
        # (miss probability ≈ 1e-3), but "all 10 trials fire" is
        # statistical, not a contract — assert a stable floor instead,
        # and keep the M4 exit criterion on p99.
        assert report.detection_probability >= 0.9
        assert report.ttfb_p99_ms < report.target_ms
        assert report.achieved is True


def test_wave_override() -> None:
    """``wave`` param overrides the config's wave (same latency floor)."""
    base = PreambleConfig(wave="zadoff-chu")
    report = compute_ttfb(base, snr_db=SNR_DB, n_trials=N_TRIALS, seed=SEED, wave="chirp")
    assert report.wave == "chirp"
    # Statistical at the 20 dB reference point (see test_all_waves_pass_m4).
    assert report.detection_probability >= 0.9


def test_buffer_and_offset_params_forwarded() -> None:
    """Explicit ``buffer_s`` / offset bounds affect the result as expected."""
    stats = measure_detection(
        CFG,
        snr_db=SNR_DB,
        n_trials=N_TRIALS,
        buffer_s=BUFFER_S,
        min_offset_frac=MIN_OFFSET_FRAC,
        max_offset_frac=MAX_OFFSET_FRAC,
        seed=SEED,
    )
    report = compute_ttfb(
        CFG,
        snr_db=SNR_DB,
        n_trials=N_TRIALS,
        buffer_s=BUFFER_S,
        min_offset_frac=MIN_OFFSET_FRAC,
        max_offset_frac=MAX_OFFSET_FRAC,
        seed=SEED,
    )
    assert report.n_trials == stats.n_trials
    assert report.ttfb_p50_ms == pytest.approx(stats.latency_p50_s * 1000.0)
    assert report.ttfb_p99_ms == pytest.approx(stats.latency_p99_s * 1000.0)


def test_invalid_offset_raises() -> None:
    """Invalid offset bounds are rejected (checked by ``measure_detection``)."""
    with pytest.raises(ValueError):
        compute_ttfb(
            CFG,
            min_offset_frac=0.5,
            max_offset_frac=0.1,
        )


def test_report_dataclass_immutable_fields() -> None:
    """TTFBReport is a plain frozen dataclass."""
    report = compute_ttfb(CFG, snr_db=SNR_DB, n_trials=N_TRIALS, seed=SEED)
    assert isinstance(report, TTFBReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.achieved = False  # frozen dataclass
