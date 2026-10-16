"""Unit tests for ``auditory_handshake.contention`` (N-device slotted ALOHA).

Covers the clean-slot probability math, the Geometric expected-slot (and cap)
behaviour, the contention TTCH, the C1 RT60-sensitivity amplification with N,
and the ``N == 1`` reduction to the single-link baseline.
"""

import importlib.util as _ilu  # noqa: E402
import sys  # noqa: E402

# ``sims`` holds underscore-prefixed simulation modules; the one exercised here
# has a dash+underscore filename (``07_n_device_contention``) that is not a
# legal Python identifier, so it cannot be imported with a plain
# ``from sims.X import``. Load it from its file path instead.
from pathlib import Path  # noqa: E402

import pytest

from auditory_handshake.contention import (
    clean_slot_probability,
    expected_slots,
    expected_turnarounds,
    ttch_s,
)
from auditory_handshake.metrics.ttch import DEFAULT_K, ttch_rtt_s

_SIM07_PATH = Path(__file__).resolve().parent.parent / "sims" / "07_n_device_contention.py"
_spec = _ilu.spec_from_file_location("sim07_n_device_contention", _SIM07_PATH)
assert _spec is not None, "sim_07 module spec must be resolvable"
_loader = _spec.loader
assert _loader is not None, "sim_07 module loader must be present"
_sim07 = _ilu.module_from_spec(_spec)
sys.modules["sim07_n_device_contention"] = _sim07
_loader.exec_module(_sim07)
contention_ttch = _sim07.contention_ttch


def test_clean_slot_probability_n1_is_one() -> None:
    assert clean_slot_probability(1, 0.2) == pytest.approx(1.0)


def test_clean_slot_probability_value() -> None:
    # (1 - 0.5) ** (3 - 1) = 0.25
    assert clean_slot_probability(3, 0.5) == pytest.approx(0.25)


def test_clean_slot_probability_decreases_with_n() -> None:
    vals = [clean_slot_probability(n, 0.2) for n in (1, 2, 3, 5, 10)]
    assert vals == sorted(vals, reverse=True)
    assert all(0.0 < v <= 1.0 for v in vals)


def test_expected_slots_n1_is_one() -> None:
    assert expected_slots(1, 0.2) == pytest.approx(1.0)


def test_expected_slots_is_geometric_mean() -> None:
    # 1 / (1 - 0.2) ** (2 - 1) = 1.25
    assert expected_slots(2, 0.2) == pytest.approx(1.25)


def test_expected_slots_cap() -> None:
    # (1/0.8) ** 19 ~ 69.4 -> capped at 20.
    assert expected_slots(20, 0.2, n_slots_max=20.0) == pytest.approx(20.0)


def test_expected_turnarounds_n1_equals_base() -> None:
    assert expected_turnarounds(1, base_turnarounds=2.0) == pytest.approx(2.0)


def test_expected_turnarounds_scales_with_n() -> None:
    assert expected_turnarounds(2, base_turnarounds=2.0, p_tx=0.2) == pytest.approx(2.5)
    assert expected_turnarounds(5, base_turnarounds=2.0, p_tx=0.2) > expected_turnarounds(
        2, base_turnarounds=2.0, p_tx=0.2
    )


def test_ttch_n1_equals_single_link_baseline() -> None:
    for rt60 in (0.0, 0.1, 0.5, 1.0):
        assert ttch_s(1, rt60) == pytest.approx(ttch_rtt_s(rt60, 0.00125, 0.03125))


def test_ttch_increases_with_n() -> None:
    rt60 = 0.5
    vals = [ttch_s(n, rt60) for n in (1, 2, 3, 5)]
    assert vals == sorted(vals)


def test_ttch_increases_with_rt60() -> None:
    for n in (1, 2, 5, 10):
        assert ttch_s(n, 0.9) > ttch_s(n, 0.1)


def test_rt60_sensitivity_amplifies_with_n() -> None:
    # dTTCH/dRT60 = T*k: grows monotonically with N while below the cap.
    def d(n: int) -> float:
        return (ttch_s(n, 0.6) - ttch_s(n, 0.4)) / 0.2

    assert d(2) > d(1) > DEFAULT_K * 2.0 * (0.99)  # d(1) ~ T*k = 2k
    assert d(1) == pytest.approx(2.0 * DEFAULT_K, rel=1e-3)
    assert d(5) > d(2)


def test_verdict_corroborated_for_all_n() -> None:
    for n in (1, 2, 3, 5, 8, 14):
        r = contention_ttch(n)
        assert r.verdict == "CORROBORATED"
        assert r.slope_ms_per_s is not None and r.slope_ms_per_s > 0.0


def test_n1_ttch_matches_baseline_at_repro_room() -> None:
    base = contention_ttch(1)
    assert base.ttch_ms_at_repro_room == pytest.approx(ttch_rtt_s(0.5, 0.00125, 0.03125) * 1000.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        clean_slot_probability(0)
    with pytest.raises(ValueError):
        clean_slot_probability(2, p_tx=0.0)
    with pytest.raises(ValueError):
        clean_slot_probability(2, p_tx=1.0)
    with pytest.raises(ValueError):
        expected_slots(2, n_slots_max=0.5)
    with pytest.raises(ValueError):
        expected_turnarounds(2, base_turnarounds=0.5)
