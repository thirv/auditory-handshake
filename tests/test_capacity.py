"""Tests for channel capacity (S3): Shannon bound + water-filling."""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.metrics.capacity import (
    shannon_capacity,
    waterfilling_capacity,
)

# --- Shannon (PLAN.md §2.1) -------------------------------------------------


def test_shannon_matches_plan_table() -> None:
    # PLAN.md §2.1 table ("~" values, so 2% tolerance).
    cases = [
        (15_000, 30, 150_000),
        (15_000, 20, 100_000),
        (15_000, 10, 52_000),
        (4_000, 20, 27_000),
    ]
    for b, snr_db, expected in cases:
        assert shannon_capacity(b, snr_db) == pytest.approx(expected, rel=0.02)


def test_shannon_monotone_in_snr() -> None:
    assert shannon_capacity(1_000, 20) > shannon_capacity(1_000, 10)


def test_shannon_invalid_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        shannon_capacity(0, 10)


# --- Water-filling -----------------------------------------------------------


def test_waterfill_flat_noise_equals_shannon() -> None:
    # K sub-bands, flat noise, unity gains => exact Shannon at the same SNR.
    k = 8
    n = 2.0
    p = 16.0
    w = 1_000.0
    res = waterfilling_capacity([n] * k, None, total_power=p, subband_width_hz=w)
    expected = shannon_capacity(k * w, 10 * np.log10(p / (k * n)))
    assert res.capacity_bits_s == pytest.approx(expected, rel=1e-9)
    assert np.allclose(res.allocated_power, p / k)


def test_waterfill_conserves_power() -> None:
    res = waterfilling_capacity([1.0, 4.0, 9.0], [1.0, 0.5, 1.0], total_power=5.0)
    assert res.allocated_power.sum() == pytest.approx(5.0)


def test_waterfill_prefers_clean_subbands() -> None:
    # High noise on sub-band 2 => less power there, some zeroed if deep fade.
    res = waterfilling_capacity([1.0, 100.0], None, total_power=2.0)
    assert res.allocated_power[0] > res.allocated_power[1]
    assert res.active_subbands == 1  # water level below N2


def test_waterfill_channel_gain_shapes_allocation() -> None:
    # Deep-fade sub-band (gain 0.1) gets nothing; clean one takes all.
    res = waterfilling_capacity([1.0, 1.0], [1.0, 0.1], total_power=10.0)
    assert res.allocated_power[1] == pytest.approx(0.0)
    assert res.allocated_power[0] == pytest.approx(10.0)


def test_waterfill_beats_uniform_split_when_noise_unflat() -> None:
    noise = [1.0, 4.0]
    p_each = 4.0  # total 8 split uniformly
    uniform = 1_000.0 * np.sum(np.log2(1.0 + np.array([p_each, p_each]) / noise))
    res = waterfilling_capacity(noise, None, total_power=8.0, subband_width_hz=1_000)
    assert res.capacity_bits_s > uniform


def test_waterfill_zero_power() -> None:
    res = waterfilling_capacity([1.0, 2.0], None, total_power=0.0)
    assert res.capacity_bits_s == 0.0
    assert res.active_subbands == 0
    assert np.all(res.allocated_power == 0)


def test_waterfill_validation() -> None:
    with pytest.raises(ValueError, match="noise"):
        waterfilling_capacity([1.0, 0.0], None, total_power=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        waterfilling_capacity([], None, total_power=1.0)
    with pytest.raises(ValueError, match="channel_gains"):
        waterfilling_capacity([1.0, 2.0], [1.0], total_power=1.0)
    with pytest.raises(ValueError, match="gains"):
        waterfilling_capacity([1.0, 2.0], [1.0, -1.0], total_power=1.0)
    with pytest.raises(ValueError, match="total_power"):
        waterfilling_capacity([1.0, 2.0], None, total_power=-1.0)
    with pytest.raises(ValueError, match="subband_width"):
        waterfilling_capacity([1.0, 2.0], None, subband_width_hz=0)
