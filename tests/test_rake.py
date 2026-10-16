"""Unit tests for the RAKE receiver (``sim.rake``).

Covers the :class:`PathTap` model, :func:`find_paths` dominant-path
extraction, and :func:`rake` maximum-ratio combining.  Self-contained (no
dependency on the repo conftest helpers) so it runs standalone.
"""

import numpy as np
import pytest

from sim.rake import PathTap, find_paths, rake


def test_path_tap_rtt_is_two_way() -> None:
    tap = PathTap(delay_samples=24, gain=1.0, fs=48_000.0)
    # one-way delay = 24 / 48k = 0.5 ms; RTT is two-way -> 1 ms.
    assert tap.delay_seconds == pytest.approx(0.5e-3)
    assert tap.rtt_s == pytest.approx(1.0e-3)
    assert tap.power == pytest.approx(1.0)


def test_path_tap_power_normalised() -> None:
    tap = PathTap(delay_samples=10, gain=0.5, fs=48_000.0)
    assert tap.power == pytest.approx(0.25)


def test_find_paths_single_direct_path() -> None:
    # A pure impulse (direct path only, at sample 7).
    rir = np.zeros(32)
    rir[7] = 1.0
    taps = find_paths(rir, threshold_db=-40.0, max_paths=8)
    assert len(taps) == 1
    assert taps[0].delay_samples == 7
    assert taps[0].gain == pytest.approx(1.0 + 0.0j)


def test_find_paths_order_and_normalisation() -> None:
    # Two strong paths: direct at 5, a 6 dB (0.5) reflection at 20.
    rir = np.zeros(64)
    rir[5] = 1.0
    rir[20] = 0.5
    taps = find_paths(rir, threshold_db=-40.0, max_paths=8)
    assert [t.delay_samples for t in taps] == [5, 20]
    assert taps[0].gain == pytest.approx(1.0)
    assert taps[1].gain == pytest.approx(0.5)


def test_find_paths_respects_threshold_db() -> None:
    # A path 12 dB below the peak (gain ~0.251) must be dropped at -10 dB.
    rir = np.zeros(64)
    rir[5] = 1.0
    rir[20] = 10.0 ** (-12 / 20.0)
    taps = find_paths(rir, threshold_db=-10.0, max_paths=8)
    assert [t.delay_samples for t in taps] == [5]


def test_find_paths_respects_max_paths() -> None:
    rir = np.zeros(128)
    for i, amp in [(5, 1.0), (20, 0.8), (40, 0.6), (60, 0.5)]:
        rir[i] = amp
    taps = find_paths(rir, threshold_db=-40.0, max_paths=2)
    assert len(taps) == 2
    # The two strongest are kept, ordered by delay.
    assert [t.delay_samples for t in taps] == [5, 20]


def test_find_paths_rejects_empty() -> None:
    assert find_paths(np.zeros(16)) == []
    assert find_paths(np.ones(16) * 0.0) == []


def test_rake_no_taps_is_identity() -> None:
    rng = np.random.default_rng(1)
    rx = rng.standard_normal(64)
    np.testing.assert_array_equal(rake(rx, []), rx)


def test_rake_collapses_reflection() -> None:
    # A direct path + a coherent reflection; RAKE combining should suppress the
    # reflection's contribution relative to the (coherently summed) direct path.
    rir = np.zeros(64)
    rir[0] = 1.0
    rir[10] = 0.8
    taps = find_paths(rir, threshold_db=-40.0, max_paths=4)

    # A unit impulse "received" -> the channel output IS the RIR.
    impulse = np.zeros(64)
    impulse[0] = 1.0

    # Without combining, the reflection shows up at sample 10.
    raw = rir
    assert raw[10] == pytest.approx(0.8)

    # With combining, the energy stacks onto the direct (sample 0) path and the
    # residual at the reflection delay is smaller.
    combined = rake(impulse, taps, shift=False)
    assert abs(combined[0]) > abs(combined[10])


def test_rake_shift_aligns_direct_path() -> None:
    rir = np.zeros(64)
    rir[0] = 1.0
    taps = find_paths(rir, threshold_db=-40.0, max_paths=4)
    impulse = np.zeros(64)
    impulse[0] = 1.0
    shifted = rake(impulse, taps, shift=True)
    assert shifted.size == 64  # trimmed back to the input length


def test_rake_deterministic_and_real() -> None:
    rng = np.random.default_rng(2)
    rir = rng.standard_normal(64)
    rir[3] = 3.0
    taps = find_paths(rir, threshold_db=-20.0, max_paths=6)
    rx = rng.standard_normal(64)
    a = rake(rx, taps)
    b = rake(rx, taps)
    assert a.dtype == np.float64
    np.testing.assert_array_equal(a, b)
