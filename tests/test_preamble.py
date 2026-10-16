"""Tests for the acoustic preamble + matched-filter detector (M4).

Covers: config validation, the normalised matched-filter single-peak
property, offset recovery in noise, and the M4 latency/FAR measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.phy import (
    PreambleConfig,
    PreambleDetector,
    generate_preamble,
    measure_detection,
)

CFG = PreambleConfig()  # defaults: zadoff-chu, 61 samples @ 48 kHz
CHIRP_CFG = PreambleConfig(wave="chirp")


def test_preamble_shape_and_amplitude() -> None:
    for cfg in (CFG, CHIRP_CFG):
        wf = generate_preamble(cfg)
        assert wf.waveform.shape == (cfg.length,)
        assert np.max(np.abs(wf.waveform)) == pytest.approx(cfg.amplitude)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        PreambleConfig(wave="mfsk")  # not a supported wave
    with pytest.raises(ValueError):
        PreambleConfig(length=62)  # not prime -> no CAZAC property
    with pytest.raises(ValueError):
        PreambleConfig(detect_threshold=1.5)  # > 1.0 is impossible
    with pytest.raises(ValueError):
        PreambleConfig(amplitude=-1.0)
    with pytest.raises(ValueError):
        PreambleConfig(wave="chirp", chirp_bandwidth=100_000.0)  # past Nyquist
    with pytest.raises(ValueError):
        PreambleConfig(wave="chirp", fc_hz=24_000.0)  # fc above Nyquist


def test_single_peak_at_alignment() -> None:
    """Normalised matched filter: 1.0 at true alignment, strictly less elsewhere."""
    for cfg in (CFG, CHIRP_CFG):
        det = PreambleDetector(cfg)
        tail = 100
        x = np.concatenate([det.preamble, np.zeros(tail)])
        env = det.matched_filter(x)
        # "valid" mode scan: len(x) - len(p) + 1 = tail + 1
        assert len(env) == tail + 1
        assert float(env[0]) == pytest.approx(1.0, abs=1e-12)
        assert int(np.argmax(env)) == 0
        assert float(np.max(env[1:])) < 0.9


def test_offset_recovery_noiseless() -> None:
    rng = np.random.default_rng(0)
    for cfg in (CFG, CHIRP_CFG):
        det = PreambleDetector(cfg)
        for _ in range(5):
            k = int(rng.integers(0, 200))
            x = np.zeros(k + cfg.length + 100)
            x[k : k + cfg.length] = det.preamble
            result = det.detect(x)
            assert result.found
            assert result.offset == k
            assert result.peak == pytest.approx(1.0, abs=1e-9)


def test_detection_in_noise_and_far() -> None:
    """High-SNR preamble is always found; noise-only trials never fire."""
    stats = measure_detection(CFG, snr_db=20.0, n_trials=100, seed=1)
    assert stats.detection_probability == 1.0
    assert stats.false_alarm_rate == 0.0
    # Latency floor is the preamble tail: (length - 1) / fs ~= 1.25 ms.
    tail_s = (CFG.length - 1) / CFG.fs
    assert tail_s <= stats.latency_p50_s <= tail_s + 1e-6
    assert stats.latency_p99_s < 8e-3  # M4 exit criterion


def test_chirp_measurement() -> None:
    stats = measure_detection(CHIRP_CFG, snr_db=20.0, n_trials=60, seed=2)
    assert stats.detection_probability == 1.0
    assert stats.latency_p99_s < 8e-3


def test_measure_detection_validation() -> None:
    with pytest.raises(ValueError):
        measure_detection(CFG, n_trials=1)
    with pytest.raises(ValueError):
        measure_detection(CFG, min_offset_frac=0.5, max_offset_frac=0.2)


def test_detector_rejects_short_input() -> None:
    det = PreambleDetector(CFG)
    with pytest.raises(ValueError):
        det.matched_filter(np.ones(10))  # shorter than the 61-sample preamble
