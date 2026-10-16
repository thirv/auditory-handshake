"""Unit tests for the canonical PLAN.md §1.4 RT60-sensitivity falsifier.

:mod:`auditory_handshake.metrics.ttch` is the *canonical* home of the PLAN.md
§1.4 check — "C1 dies if TTCH is statistically insensitive to RT60".
``metrics/two_way.py`` and ``sims/09_rt60_sensitivity.py`` build on it, and
``tests/test_two_way.py`` only exercises the two-way wrapper's *delegation*.
This module pins the three pure building blocks directly:

- :func:`ttch_rtt_s` — the PLAN.md §2.4 round-trip turnaround model,
- :func:`regress_rt60_sensitivity` — the OLS fit of a measured per-room
  latency on RT60, and its verdict ladder, and
- :func:`classify_regime` — the PLAN.md §2.4 / §3.1 win/lose regime split.

Pure (no I/O, no room synthesis); the hand-written datasets keep each verdict
bucket fully deterministic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from auditory_handshake import metrics
from auditory_handshake.metrics import ttch
from auditory_handshake.metrics.ttch import (
    RT60Sensitivity,
    classify_regime,
    regress_rt60_sensitivity,
    ttch_rtt_s,
)

# --- ttch_rtt_s (PLAN.md §2.4 round-trip model) -------------------------


def test_ttch_rtt_s_matches_the_plan_formula() -> None:
    # ``turnarounds · (T_preamble + T_frame + k·RT60) + T_proc``.
    rt60_s, t_preamble_s, t_frame_s = 0.30, 0.001, 0.03
    t_proc_s, k, turnarounds = 0.010, 0.35, 2
    expected = turnarounds * (t_preamble_s + t_frame_s + k * rt60_s) + t_proc_s
    assert ttch_rtt_s(
        rt60_s, t_preamble_s, t_frame_s, t_proc_s=t_proc_s, k=k, turnarounds=turnarounds
    ) == pytest.approx(expected)


def test_ttch_rtt_s_is_monotone_in_rt60() -> None:
    # The C1 premise: more reverberation -> longer turnaround time.
    assert ttch_rtt_s(0.5, 0.001, 0.03) > ttch_rtt_s(0.1, 0.001, 0.03)


def test_ttch_rtt_s_is_monotone_in_k() -> None:
    # k is the PLAN.md §2.4 echo-tail constant (range [0.2, 0.5]).
    assert ttch_rtt_s(0.3, 0.001, 0.03, k=0.5) > ttch_rtt_s(0.3, 0.001, 0.03, k=0.2)


def test_ttch_rtt_s_is_monotone_in_turnarounds() -> None:
    assert ttch_rtt_s(0.3, 0.001, 0.03, turnarounds=3) > ttch_rtt_s(0.3, 0.001, 0.03, turnarounds=1)


def test_ttch_rtt_s_rejects_a_negative_stage() -> None:
    with pytest.raises(ValueError):
        ttch_rtt_s(0.3, -0.001, 0.03)


def test_ttch_rtt_s_rejects_negative_frame_or_proc() -> None:
    with pytest.raises(ValueError):
        ttch_rtt_s(0.3, 0.001, -0.03)
    with pytest.raises(ValueError):
        ttch_rtt_s(0.3, 0.001, 0.03, t_proc_s=-0.01)


def test_ttch_rtt_s_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        ttch_rtt_s(0.3, 0.001, 0.03, k=0.0)


def test_ttch_rtt_s_rejects_fewer_than_one_turnaround() -> None:
    with pytest.raises(ValueError):
        ttch_rtt_s(0.3, 0.001, 0.03, turnarounds=0)


# --- classify_regime (PLAN.md §2.4 / §3.1 win/lose split) ---------------


def test_classify_regime_win_lose_split() -> None:
    # Default threshold is 0.35 s: sub-meter / cabin / docking win.
    assert classify_regime(0.102) == "win"
    assert classify_regime(0.711) == "lose"


def test_classify_regime_boundary_is_a_loss() -> None:
    # The win test is strict (RT60 < threshold), so the boundary loses.
    assert classify_regime(0.35) == "lose"


def test_classify_regime_honours_a_custom_threshold() -> None:
    assert classify_regime(0.2, threshold_s=0.3) == "win"
    assert classify_regime(0.4, threshold_s=0.3) == "lose"


def test_classify_regime_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        classify_regime(-1.0)
    with pytest.raises(ValueError):
        classify_regime(0.5, threshold_s=0.0)


# --- regress_rt60_sensitivity (PLAN.md §1.4 verdict ladder) -------------


def test_regress_corroborated_on_a_strong_positive_slope() -> None:
    rt60 = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    latency = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # perfectly linear, slope 2
    got = regress_rt60_sensitivity(rt60, latency)
    assert got.verdict == "CORROBORATED"
    assert got.slope == pytest.approx(2.0)
    assert got.intercept == pytest.approx(1.0)
    assert got.pearson_r == pytest.approx(1.0)
    assert got.t_stat is not None and got.t_stat > 0


def test_regress_weakly_corroborated_on_a_moderate_slope() -> None:
    # Pearson r ~0.78 at n=6 lands in the [weak_t, strong_t) = [2, 3) t-band.
    rt60 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    latency = [1.0, 2.0, 1.0, 3.0, 2.0, 4.0]
    got = regress_rt60_sensitivity(rt60, latency)
    assert got.verdict == "WEAKLY_CORROBORATED"
    assert got.slope > 0


def test_regress_at_risk_on_a_weak_positive_slope() -> None:
    # Positive but weak (r ~0.49 at n=6): slope > 0, |t| below weak_t.
    rt60 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    latency = [1.0, 2.0, 1.0, 1.0, 2.0, 2.0]
    got = regress_rt60_sensitivity(rt60, latency)
    assert got.verdict == "AT_RISK"
    assert got.slope > 0


def test_regress_contradicted_on_a_negative_slope() -> None:
    rt60 = [0.1, 0.2, 0.3, 0.4, 0.5]
    latency = [10.0, 9.0, 8.0, 7.0, 6.0]  # decreasing -> negative slope
    got = regress_rt60_sensitivity(rt60, latency)
    assert got.verdict == "CONTRADICTED"
    assert got.slope < 0
    assert got.pearson_r < 0


def test_regress_inconclusive_with_two_samples() -> None:
    got = regress_rt60_sensitivity([0.10, 0.50], [0.20, 0.60])
    assert got.verdict == "INCONCLUSIVE"
    assert got.t_stat is None
    assert got.n == 2


# --- regress_rt60_sensitivity input guards -------------------------------


def test_regress_rejects_non_finite_rt60() -> None:
    with pytest.raises(ValueError):
        regress_rt60_sensitivity([0.10, float("inf"), 0.30, 0.40, 0.50], [1.0] * 5)


def test_regress_rejects_non_finite_latency() -> None:
    with pytest.raises(ValueError):
        regress_rt60_sensitivity([0.10, 0.20, 0.30, 0.40, 0.50], [1.0, float("nan"), 3.0, 4.0, 5.0])


def test_regress_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        regress_rt60_sensitivity([0.10, 0.20, 0.30], [1.0, 2.0])


def test_regress_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError):
        regress_rt60_sensitivity([0.10], [1.0])


def test_regress_rejects_zero_variance_rt60() -> None:
    with pytest.raises(ValueError):
        regress_rt60_sensitivity([0.30] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_regress_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        regress_rt60_sensitivity(np.array([[0.10, 0.20, 0.30]]), [1.0, 2.0, 3.0])


# --- RT60Sensitivity is immutable ---------------------------------------


def test_rt60_sensitivity_result_is_frozen() -> None:
    got: RT60Sensitivity = regress_rt60_sensitivity([0.10, 0.20, 0.30], [1.0, 2.0, 3.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.slope = 0.0  # type: ignore[misc]  # frozen dataclass -> must raise


# --- package re-exports (PEP 562 lazy map) -------------------------------


def test_package_reexports_the_ttch_names() -> None:
    assert metrics.ttch is ttch
    assert metrics.ttch_rtt_s is ttch_rtt_s
    assert metrics.classify_regime is classify_regime
    assert metrics.regress_rt60_sensitivity is regress_rt60_sensitivity
