"""Unit tests for S6 acoustic link budget (``metrics/link_budget.py``).

Covers the free-space propagation law, SNR, noise-floor power-sum, capacity
delegation to the S3 ``shannon_capacity`` primitive, monotonic decrease of
capacity with distance, and the config-driven inputs (scenario noise floor,
device TX level, default bandwidth).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from auditory_handshake.config import (
    load_defaults,
    load_device_preset,
    load_scenario,
)
from auditory_handshake.metrics.capacity import shannon_capacity
from auditory_handshake.metrics.link_budget import (
    channel_capacity,
    free_space_spl,
    link_budget,
    link_budget_sweep,
    link_snr,
    noise_floor_db,
)


# --------------------------------------------------------------------------- #
# free-space propagation
# --------------------------------------------------------------------------- #
def test_free_space_spl_reference_distance():
    # No loss at the reference distance.
    assert free_space_spl(90.0, 1.0) == pytest.approx(90.0)


def test_free_space_spl_doubling_drops_6_db():
    # Inverse-square: doubling distance drops SPL by 20*log10(2) ≈ 6.02 dB.
    drop = free_space_spl(90.0, 1.0) - free_space_spl(90.0, 2.0)
    assert drop == pytest.approx(20.0 * math.log10(2.0), rel=1e-9)
    assert drop == pytest.approx(6.0206, abs=1e-3)


def test_free_space_spl_arbitrary_distance():
    # d_ref = 1 m, 10 m -> 90 - 20*log10(10) = 70 dB.
    assert free_space_spl(90.0, 10.0) == pytest.approx(70.0)


def test_free_space_spl_custom_reference():
    # With d_ref = 2 m, at d = 2 m there is still no loss.
    assert free_space_spl(90.0, 2.0, reference_distance_m=2.0) == pytest.approx(90.0)


def test_free_space_spl_invalid_distance():
    with pytest.raises(ValueError):
        free_space_spl(90.0, 0.0)
    with pytest.raises(ValueError):
        free_space_spl(90.0, -3.0)
    with pytest.raises(ValueError):
        free_space_spl(90.0, 5.0, reference_distance_m=0.0)


# --------------------------------------------------------------------------- #
# SNR
# --------------------------------------------------------------------------- #
def test_link_snr_basic():
    assert link_snr(50.0, 40.0) == pytest.approx(10.0)
    assert link_snr(40.0, 40.0) == pytest.approx(0.0)
    assert link_snr(30.0, 40.0) == pytest.approx(-10.0)


# --------------------------------------------------------------------------- #
# noise-floor combination
# --------------------------------------------------------------------------- #
def test_noise_floor_db_single_and_empty():
    assert noise_floor_db([38.0]) == pytest.approx(38.0)
    assert noise_floor_db([]) == pytest.approx(0.0)


def test_noise_floor_db_power_sum():
    # Two equal 40 dB sources -> 40 + 3.01 = 43.01 dB.
    assert noise_floor_db([40.0, 40.0]) == pytest.approx(10 * math.log10(2 * 10**4.0))
    # Two unequal sources -> exact power sum.
    expected = 10 * math.log10(10**4.2 + 10**5.5)
    assert noise_floor_db([42.0, 55.0]) == pytest.approx(expected)
    # Result is at least the loudest source.
    assert noise_floor_db([42.0, 55.0]) > 55.0


# --------------------------------------------------------------------------- #
# capacity (delegates to S3 shannon_capacity)
# --------------------------------------------------------------------------- #
def test_channel_capacity_delegates_to_shannon():
    for snr in (-10.0, 0.0, 10.0, 20.0, 40.0):
        assert channel_capacity(snr, 4000.0) == pytest.approx(shannon_capacity(4000.0, snr))


def test_channel_capacity_invalid_bandwidth():
    with pytest.raises(ValueError):
        channel_capacity(10.0, 0.0)
    with pytest.raises(ValueError):
        channel_capacity(10.0, -5.0)


# --------------------------------------------------------------------------- #
# link_budget (single point) consistency + monotonicity
# --------------------------------------------------------------------------- #
def test_link_budget_consistency():
    budget = link_budget(distance_m=1.0, tx_spl_db=100.0, noise_floor_db=40.0, bandwidth_hz=4000.0)
    assert budget["distance_m"] == pytest.approx(1.0)
    assert budget["spl_rx_db"] == pytest.approx(100.0)
    assert budget["snr_db"] == pytest.approx(60.0)
    assert budget["capacity_bps"] == pytest.approx(shannon_capacity(4000.0, 60.0))


def test_link_budget_monotonic_decrease_with_distance():
    budgets = [
        link_budget(d, tx_spl_db=100.0, noise_floor_db=40.0, bandwidth_hz=4000.0)
        for d in (0.5, 1.0, 2.0, 5.0, 10.0)
    ]
    spl = [b["spl_rx_db"] for b in budgets]
    snr = [b["snr_db"] for b in budgets]
    cap = [b["capacity_bps"] for b in budgets]
    assert all(a > b for a, b in zip(spl, spl[1:], strict=False))
    assert all(a > b for a, b in zip(snr, snr[1:], strict=False))
    assert all(a > b for a, b in zip(cap, cap[1:], strict=False))


# --------------------------------------------------------------------------- #
# link_budget_sweep
# --------------------------------------------------------------------------- #
def test_link_budget_sweep_shapes_and_monotonic():
    d = np.linspace(0.5, 10.0, 20)
    sw = link_budget_sweep(d, tx_spl_db=100.0, noise_floor_db=40.0, bandwidth_hz=4000.0)
    assert sw["distances_m"].shape == d.shape
    assert len(sw["spl_rx_db"]) == 20
    assert len(sw["snr_db"]) == 20
    assert len(sw["capacity_bps"]) == 20
    assert all(a > b for a, b in zip(sw["capacity_bps"], sw["capacity_bps"][1:], strict=False))


def test_link_budget_sweep_matches_pointwise():
    d = np.array([1.0, 3.0, 7.0])
    sw = link_budget_sweep(d, tx_spl_db=100.0, noise_floor_db=40.0, bandwidth_hz=4000.0)
    for i, dist in enumerate(d):
        point = link_budget(dist, tx_spl_db=100.0, noise_floor_db=40.0, bandwidth_hz=4000.0)
        assert sw["spl_rx_db"][i] == pytest.approx(point["spl_rx_db"])
        assert sw["snr_db"][i] == pytest.approx(point["snr_db"])
        assert sw["capacity_bps"][i] == pytest.approx(point["capacity_bps"])


def test_link_budget_sweep_invalid_distance():
    with pytest.raises(ValueError):
        link_budget_sweep(
            [0.5, 0.0, 2.0], tx_spl_db=100.0, noise_floor_db=40.0, bandwidth_hz=4000.0
        )


# --------------------------------------------------------------------------- #
# config-driven inputs (scenario noise floor, device TX level, bandwidth)
# --------------------------------------------------------------------------- #
def test_config_driven_noise_floor_from_scenario():
    quiet = load_scenario("quiet_office")
    noisy = load_scenario("conference_call")
    nf_quiet = noise_floor_db([ns.level_db_spl for ns in quiet.noise_sources])
    nf_noisy = noise_floor_db([ns.level_db_spl for ns in noisy.noise_sources])
    # quiet_office is a single 38 dB source.
    assert nf_quiet == pytest.approx(38.0)
    # Noisy office is louder than the quiet one.
    assert nf_noisy > nf_quiet


def test_tx_level_from_device_preset_is_positive():
    tx = load_device_preset("small_loudspeaker")
    assert tx.max_spl_db > 0.0


def test_defaults_bandwidth_positive():
    defaults = load_defaults()
    band = defaults["audio_bands_hz"]["css"]
    assert band[1] - band[0] > 0.0
