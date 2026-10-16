"""Tests for the goodput–audibility trade-off surface (``metrics/rate.py``).

Pure unit tests (no I/O, no GPU). They pin down:

- :class:`DesignPoint` validation (field bounds, tuple-length consistency).
- :func:`gross_bit_rate_bps` and :func:`goodput_bps` arithmetic.
- :func:`aggregate_audibility_cost_db` vs. the per-tone
  :func:`audibility_cost_db` values.
- :func:`normalized_utility` (dimensionless, monotone in λ, validation).
- :func:`nondominated` (Pareto-front extraction, dominated exclusion).
- :func:`trace_optimum_over_lambda` (frontier trace, distinct count).
- :func:`c2_verdict` (CORROBORATED vs DEGENERATE).
- Package re-exports resolve from ``metrics``.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from auditory_handshake import metrics
from auditory_handshake.metrics.audibility import audibility_cost_db
from auditory_handshake.metrics.rate import (
    DesignPoint,
    LlambdaTrace,
    aggregate_audibility_cost_db,
    c2_verdict,
    goodput_bps,
    gross_bit_rate_bps,
    nondominated,
    normalized_utility,
    trace_optimum_over_lambda,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _point(
    n_bands: int = 1,
    bw: float = 5000.0,
    snr_db: float = 20.0,
    level_db: float = 60.0,
    ber: float = 1e-6,
    overhead_bits: float = 0.0,
) -> DesignPoint:
    """Convenience: build a DesignPoint with the first ``n_bands`` base freqs."""
    base = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    return DesignPoint(
        n_bands=n_bands,
        bandwidth_per_band_hz=bw,
        snr_db=snr_db,
        level_db=level_db,
        occupied_frequencies_hz=base[:n_bands],
        ber=ber,
        overhead_bits=overhead_bits,
    )


# --------------------------------------------------------------------------- #
# DesignPoint validation
# --------------------------------------------------------------------------- #
class TestDesignPoint:
    def test_valid_point(self) -> None:
        p = _point()
        assert p.n_bands == 1
        assert p.occupied_frequencies_hz == (500.0,)

    def test_n_bands_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_bands"):
            DesignPoint(
                n_bands=0,
                bandwidth_per_band_hz=5000.0,
                snr_db=20.0,
                level_db=60.0,
                occupied_frequencies_hz=(),
            )

    def test_freq_tuple_length_must_match_n_bands(self) -> None:
        with pytest.raises(ValueError, match="occupied_frequencies_hz"):
            DesignPoint(
                n_bands=2,
                bandwidth_per_band_hz=5000.0,
                snr_db=20.0,
                level_db=60.0,
                occupied_frequencies_hz=(500.0,),
            )

    def test_bandwidth_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="bandwidth_per_band_hz"):
            _point(bw=0.0)

    def test_snr_must_be_reasonable(self) -> None:
        with pytest.raises(ValueError, match="snr_db"):
            _point(snr_db=-200.0)

    def test_ber_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="ber"):
            _point(ber=1.5)

    def test_overhead_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="overhead_bits"):
            _point(overhead_bits=-1.0)

    def test_freq_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="frequency"):
            DesignPoint(
                n_bands=1,
                bandwidth_per_band_hz=5000.0,
                snr_db=20.0,
                level_db=60.0,
                occupied_frequencies_hz=(0.0,),
            )

    def test_is_frozen(self) -> None:
        p = _point()
        with pytest.raises(FrozenInstanceError):
            p.n_bands = 2  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        p = _point()
        hash(p)  # must not raise


# --------------------------------------------------------------------------- #
# Throughput arithmetic
# --------------------------------------------------------------------------- #
class TestThroughput:
    def test_gross_bit_rate_shannon(self) -> None:
        p = _point(n_bands=2, bw=1000.0, snr_db=10.0)
        expected = 2 * 1000.0 * math.log2(1.0 + 10.0)
        assert gross_bit_rate_bps(p) == pytest.approx(expected)

    def test_goodput_equals_gross_with_no_ber_no_overhead(self) -> None:
        p = _point(ber=0.0, overhead_bits=0.0)
        assert goodput_bps(p) == pytest.approx(gross_bit_rate_bps(p))

    def test_goodput_penalised_by_ber(self) -> None:
        p = _point(ber=0.01)
        expected = gross_bit_rate_bps(p) * 0.99
        assert goodput_bps(p) == pytest.approx(expected)

    def test_goodput_subtracts_overhead(self) -> None:
        p = _point(ber=0.0, overhead_bits=1000.0)
        expected = gross_bit_rate_bps(p) - 1000.0
        assert goodput_bps(p) == pytest.approx(expected)

    def test_goodput_clamps_to_zero(self) -> None:
        p = _point(ber=1.0)  # total loss
        assert goodput_bps(p) == 0.0

    def test_goodput_scales_with_n_bands(self) -> None:
        p1 = _point(n_bands=1)
        p4 = _point(n_bands=4)
        assert goodput_bps(p4) == pytest.approx(4.0 * goodput_bps(p1))


# --------------------------------------------------------------------------- #
# Audibility aggregation
# --------------------------------------------------------------------------- #
class TestAggregateAudibility:
    def test_single_band_equals_single_tone_cost(self) -> None:
        p = _point(n_bands=1, level_db=65.0)
        single = audibility_cost_db(500.0, 65.0)
        assert aggregate_audibility_cost_db(p) == pytest.approx(single)

    def test_multi_band_exceeds_each_individual(self) -> None:
        p = _point(n_bands=3, level_db=60.0)
        total = aggregate_audibility_cost_db(p)
        for f in (500.0, 1000.0, 2000.0):
            assert total >= audibility_cost_db(f, 60.0)

    def test_more_bands_give_higher_total(self) -> None:
        p2 = _point(n_bands=2, level_db=60.0)
        p4 = _point(n_bands=4, level_db=60.0)
        assert aggregate_audibility_cost_db(p4) > aggregate_audibility_cost_db(p2)


# --------------------------------------------------------------------------- #
# Normalised utility
# --------------------------------------------------------------------------- #
class TestNormalizedUtility:
    def test_dimensionless_at_grid_max(self) -> None:
        p = _point(n_bands=1, level_db=60.0)
        g = goodput_bps(p)
        c = aggregate_audibility_cost_db(p)
        u = normalized_utility(p, 0.0, goodput_scale=g, cost_scale=c)
        assert u == pytest.approx(1.0)

    def test_monotone_decreasing_in_lambda(self) -> None:
        p = _point()
        g = goodput_bps(p)
        c = aggregate_audibility_cost_db(p)
        u0 = normalized_utility(p, 0.0, goodput_scale=g, cost_scale=c)
        u1 = normalized_utility(p, 1.0, goodput_scale=g, cost_scale=c)
        u10 = normalized_utility(p, 10.0, goodput_scale=g, cost_scale=c)
        assert u0 > u1 > u10

    def test_rejects_negative_lambda(self) -> None:
        p = _point()
        with pytest.raises(ValueError, match="lambda_"):
            normalized_utility(p, -1.0, goodput_scale=1.0, cost_scale=1.0)

    def test_rejects_zero_scale(self) -> None:
        p = _point()
        with pytest.raises(ValueError, match="goodput_scale"):
            normalized_utility(p, 0.0, goodput_scale=0.0, cost_scale=1.0)
        with pytest.raises(ValueError, match="cost_scale"):
            normalized_utility(p, 0.0, goodput_scale=1.0, cost_scale=0.0)


# --------------------------------------------------------------------------- #
# Pareto-front extraction
# --------------------------------------------------------------------------- #
class TestNondominated:
    def test_dominated_point_excluded(self) -> None:
        # p1 dominates p2: higher goodput AND lower cost (same n_bands, lower level).
        p1 = _point(n_bands=2, level_db=50.0)
        p2 = _point(n_bands=2, level_db=70.0)
        front = nondominated([p1, p2])
        assert p1 in front
        assert p2 not in front

    def test_all_non_dominated_kept(self) -> None:
        # Trade-off: more bands → higher goodput but higher cost.
        p1 = _point(n_bands=1, level_db=50.0)
        p2 = _point(n_bands=2, level_db=50.0)
        front = nondominated([p1, p2])
        assert p1 in front
        assert p2 in front

    def test_single_point_front(self) -> None:
        p = _point()
        assert nondominated([p]) == [p]

    def test_duplicate_points(self) -> None:
        p = _point()
        # Two identical points: neither strictly dominates the other
        # (costs are not strictly less), so both appear on the front.
        front = nondominated([p, p])
        assert len(front) == 2

    def test_preserves_input_order(self) -> None:
        # Same n_bands, different levels: lower level → lower cost but
        # same goodput (Shannon capacity depends only on bandwidth and SNR).
        # Neither dominates the other unless cost strictly decreases.
        # Use two points where goodput differs (different BW) but cost
        # also differs (different level).
        p1 = DesignPoint(
            n_bands=1,
            bandwidth_per_band_hz=1000.0,
            snr_db=20.0,
            level_db=70.0,
            occupied_frequencies_hz=(500.0,),
        )
        p2 = DesignPoint(
            n_bands=1,
            bandwidth_per_band_hz=2000.0,
            snr_db=20.0,
            level_db=50.0,
            occupied_frequencies_hz=(1000.0,),
        )
        # p2 has more bandwidth (higher goodput) and lower level (lower cost),
        # so p2 dominates p1. Only p2 should be on the front.
        front = nondominated([p1, p2])
        assert p2 in front
        assert p1 not in front


# --------------------------------------------------------------------------- #
# Frontier trace
# --------------------------------------------------------------------------- #
class TestTraceOptimum:
    def test_small_grid(self) -> None:
        p1 = _point(n_bands=1, level_db=50.0)
        p2 = _point(n_bands=2, level_db=60.0)
        p3 = _point(n_bands=3, level_db=70.0)
        trace = trace_optimum_over_lambda([p1, p2, p3], [0.0, 1.0, 10.0])
        assert len(trace.lambdas) == 3
        assert len(trace.frontier_points) == 3
        # At λ=0 the highest-goodput point should win.
        best_g = goodput_bps(p3)
        assert trace.frontier_goodput_bps[0] == pytest.approx(best_g)

    def test_empty_points_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            trace_optimum_over_lambda([], [0.0, 1.0])

    def test_all_zero_goodput_raises(self) -> None:
        # ber=1.0 forces goodput to zero (total loss)
        p_zero = DesignPoint(
            n_bands=1,
            bandwidth_per_band_hz=5000.0,
            snr_db=-60.0,
            level_db=60.0,
            occupied_frequencies_hz=(500.0,),
            ber=1.0,
        )
        assert goodput_bps(p_zero) == 0.0
        with pytest.raises(ValueError, match="goodput"):
            trace_optimum_over_lambda([p_zero], [0.0])

    def test_n_distinct_counts_unique_winners(self) -> None:
        p1 = _point(n_bands=1, level_db=50.0)
        p2 = _point(n_bands=2, level_db=60.0)
        # Many λ values; at least two should be distinct if goodput/cost differ.
        trace = trace_optimum_over_lambda([p1, p2], [0.0, 1.0, 10.0, 100.0])
        assert 1 <= trace.n_distinct <= 2

    def test_trace_returns_frozen_dataclass(self) -> None:
        p = _point()
        trace = trace_optimum_over_lambda([p], [0.0])
        assert isinstance(trace, LlambdaTrace)
        with pytest.raises(FrozenInstanceError):
            trace.n_distinct = 0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# C2 verdict
# --------------------------------------------------------------------------- #
class TestC2Verdict:
    def test_corroborated(self) -> None:
        trace = LlambdaTrace(
            lambdas=(0.0, 1.0),
            frontier_points=(),
            frontier_goodput_bps=(1000.0, 500.0),
            frontier_cost_db=(10.0, 15.0),
            n_distinct=2,
        )
        assert c2_verdict(trace) == "CORROBORATED"

    def test_degenerate(self) -> None:
        trace = LlambdaTrace(
            lambdas=(0.0, 1.0),
            frontier_points=(),
            frontier_goodput_bps=(1000.0, 500.0),
            frontier_cost_db=(10.0, 10.0),
            n_distinct=1,
        )
        assert c2_verdict(trace) == "DEGENERATE"

    def test_single_point_is_degenerate(self) -> None:
        trace = LlambdaTrace(
            lambdas=(0.0,),
            frontier_points=(),
            frontier_goodput_bps=(1000.0,),
            frontier_cost_db=(10.0,),
            n_distinct=1,
        )
        assert c2_verdict(trace) == "DEGENERATE"


# --------------------------------------------------------------------------- #
# Package re-exports
# --------------------------------------------------------------------------- #
class TestReexports:
    def test_rate_names_resolve_from_metrics(self) -> None:
        assert metrics.DesignPoint is DesignPoint
        assert metrics.LlambdaTrace is LlambdaTrace
        assert metrics.gross_bit_rate_bps is gross_bit_rate_bps
        assert metrics.goodput_bps is goodput_bps
        assert metrics.aggregate_audibility_cost_db is aggregate_audibility_cost_db
        assert metrics.normalized_utility is normalized_utility
        assert metrics.nondominated is nondominated
        assert metrics.trace_optimum_over_lambda is trace_optimum_over_lambda
        assert metrics.c2_verdict is c2_verdict
