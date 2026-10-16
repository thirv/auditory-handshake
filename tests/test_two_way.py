"""Tests for the two-way handshake round-trip metric (``metrics/two_way.py``).

Pure unit tests (no I/O, no room synthesis). They pin down the leg arithmetic,
the monotonicity of :func:`rtt_s` in the reverb-dependent guard term, the thin
delegation of :func:`regress_ttch_vs_rt60` to the canonical PLAN.md §1.4
falsifier, and that the package re-exports resolve.
"""

from __future__ import annotations

import pytest

from auditory_handshake import metrics
from auditory_handshake.metrics import ttch
from auditory_handshake.metrics.two_way import TwoWayLeg, regress_ttch_vs_rt60, rtt_s


def _leg(t_guard_s: float = 0.10, decoded: bool = True) -> TwoWayLeg:
    """A leg with fixed stage times and a tunable guard (the C1 reverb term)."""
    return TwoWayLeg(
        t_tx_s=0.032,
        t_guard_s=t_guard_s,
        t_detect_s=0.020,
        t_proc_s=0.010,
        decoded=decoded,
    )


def test_rtt_s_is_the_sum_of_both_legs() -> None:
    fwd = TwoWayLeg(t_tx_s=0.030, t_guard_s=0.100, t_detect_s=0.020, t_proc_s=0.010, decoded=True)
    rev = TwoWayLeg(t_tx_s=0.028, t_guard_s=0.100, t_detect_s=0.020, t_proc_s=0.010, decoded=True)
    expected = 0.030 + 0.100 + 0.020 + 0.010 + 0.028 + 0.100 + 0.020 + 0.010
    assert rtt_s(fwd, rev) == pytest.approx(expected)


def test_rtt_s_grows_monotonically_with_guard() -> None:
    # The guard is the C1 reverb-dependent term: more reverb -> longer RTT.
    assert rtt_s(_leg(0.20), _leg(0.20)) > rtt_s(_leg(0.05), _leg(0.05))


def test_decoded_flag_does_not_change_rtt() -> None:
    # A failed leg still consumed transmit+guard+detect time.
    assert rtt_s(_leg(decoded=True), _leg(decoded=True)) == rtt_s(
        _leg(decoded=False), _leg(decoded=False)
    )


def test_rtt_s_rejects_a_negative_stage() -> None:
    bad = TwoWayLeg(t_tx_s=-0.01, t_guard_s=0.0, t_detect_s=0.0, t_proc_s=0.0, decoded=True)
    with pytest.raises(ValueError):
        rtt_s(bad, _leg())


def test_regress_delegates_to_the_canonical_falsifier() -> None:
    rt60 = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
    ttch_ = [0.20 + 0.5 * r for r in rt60]  # strictly increasing -> positive slope
    got = regress_ttch_vs_rt60(rt60, ttch_)
    ref = ttch.regress_rt60_sensitivity(rt60, ttch_)
    assert got.verdict == ref.verdict
    assert got.slope == pytest.approx(ref.slope)
    assert got.n == ref.n


def test_regress_is_inconclusive_with_too_few_samples() -> None:
    got = regress_ttch_vs_rt60([0.10, 0.50], [0.20, 0.60])
    assert got.verdict == "INCONCLUSIVE"


def test_regress_is_contradicted_on_a_negative_slope() -> None:
    rt60 = [0.10, 0.30, 0.50, 0.70, 1.00]
    ttch_ = [0.50, 0.50, 0.40, 0.30, 0.20]  # decreasing -> negative slope
    got = regress_ttch_vs_rt60(rt60, ttch_)
    assert got.slope < 0
    assert got.verdict == "CONTRADICTED"


def test_package_reexports_the_two_way_names() -> None:
    assert metrics.TwoWayLeg is TwoWayLeg
    assert metrics.rtt_s is rtt_s
    assert metrics.regress_ttch_vs_rt60 is regress_ttch_vs_rt60
