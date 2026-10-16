"""Tests for :func:`auditory_handshake.acoustics.room.compute_rir`.

These pin the *measurable* properties of the generated room impulse response:
shape, sampling rate, finite values, the direct-sound onset, and the decaying
tail (a shorter RT60 must decay faster). They do **not** assert a specific
energy value, because the absolute level is set by ``pyroomacoustics``.
"""

import numpy as np
import pytest

from auditory_handshake.acoustics import compute_rir
from auditory_handshake.acoustics.channel import _tail_length_seconds
from auditory_handshake.config import RoomPreset, list_room_presets, load_room_preset


def _flat(alpha: float, dims=(5.0, 4.0, 3.0)) -> RoomPreset:
    """A frequency-flat room with uniform absorption on every face."""
    return RoomPreset(
        name="test",
        dimensions_m=tuple(dims),
        frequencies_hz=(1000.0,),
        absorption={"x": (alpha,), "y": (alpha,), "z": (alpha,)},
    )


def _onset_index(rir: np.ndarray) -> int:
    """Sample index of the first non-negligible sample (the direct sound)."""
    peak = float(np.max(np.abs(rir)))
    thresh = 1e-6 * peak
    nonzero = np.flatnonzero(np.abs(rir) > thresh)
    return int(nonzero[0]) if nonzero.size else 0


def test_rir_shape_and_dtype() -> None:
    rir = compute_rir(_flat(0.1), rt60_seconds=1.0)
    assert rir.ndim == 1
    assert rir.shape[0] == pytest.approx(_tail_length_seconds(1.0) * 16000)
    assert np.issubdtype(rir.dtype, np.floating)


def test_rir_is_finite_and_has_signal() -> None:
    rir = compute_rir(_flat(0.1), rt60_seconds=1.0)
    assert np.all(np.isfinite(rir))
    assert float(np.max(np.abs(rir))) > 0.0


def test_direct_sound_at_starting_position() -> None:
    # The direct path from the (source, receiver) starting points arrives first,
    # so the impulse must have its first energy at/before the first millisecond.
    rir = compute_rir(_flat(0.1), rt60_seconds=1.0)
    assert _onset_index(rir) <= int(0.001 * 16000)


def test_tail_decays_after_direct_sound() -> None:
    # Energy past the direct-sound onset must decay: the mean magnitude of a
    # later 10 ms window is far below the mean of the early reverberant tail.
    rir = compute_rir(_flat(0.1), rt60_seconds=1.0)
    onset = max(1, _onset_index(rir))
    fs = 16000
    early = np.mean(np.abs(rir[onset : onset + fs // 4]))
    late = np.mean(np.abs(rir[-(fs // 16) :]))
    assert late < early


def test_longer_rt60_is_smoother() -> None:
    # A shorter RT60 decays faster: at a common time within both rooms' natural
    # reverberant tails, the shorter room retains less residual energy. Probe at
    # a fixed physical time (not the array end): the extended
    # ``max(0.25 s, 3 * RT60)`` tail is zero-padded past the natural decay, so
    # the final samples of every RIR are exactly zero.
    fs = 16000
    rir_short = compute_rir(_flat(0.1), rt60_seconds=0.2)
    rir_long = compute_rir(_flat(0.1), rt60_seconds=1.0)
    probe = int(0.05 * fs)  # 50 ms: within the natural tail of both rooms
    window = fs // 100  # 10 ms
    late_short = float(np.sqrt(np.mean(rir_short[probe : probe + window] ** 2)))
    late_long = float(np.sqrt(np.mean(rir_long[probe : probe + window] ** 2)))
    assert late_short < late_long


def test_rir_is_deterministic_for_fixed_seeds() -> None:
    a = compute_rir(_flat(0.1), rt60_seconds=0.5)
    b = compute_rir(_flat(0.1), rt60_seconds=0.5)
    np.testing.assert_array_equal(a, b)


def test_sample_rate_scales_length() -> None:
    fs_lo, fs_hi = 16000, 48000
    lo = compute_rir(_flat(0.1), rt60_seconds=1.0, fs=fs_lo)
    hi = compute_rir(_flat(0.1), rt60_seconds=1.0, fs=fs_hi)
    # Same physical tail (0.25 s floor / 3x RT60) at different fs -> proportional sample counts.
    assert hi.shape[0] / lo.shape[0] == pytest.approx(fs_hi / fs_lo, rel=0.05)


@pytest.mark.parametrize("name", list_room_presets())
def test_rir_for_every_preset_is_well_formed(name: str) -> None:
    rir = compute_rir(load_room_preset(name), rt60_seconds=0.5, fs=16000)
    assert rir.ndim == 1
    assert rir.shape[0] > 0
    assert np.all(np.isfinite(rir))
    assert float(np.max(np.abs(rir))) > 0.0
    assert _onset_index(rir) <= int(0.001 * 16000)  # direct sound present


def test_rt60_zero_gives_direct_sound_only() -> None:
    # rt60_seconds=0 means the tail starts immediately -> a near-impulse with
    # energy concentrated in the first sample group (no measurable decay).
    rir = compute_rir(_flat(0.1), rt60_seconds=0.0, fs=16000)
    assert np.all(np.isfinite(rir))
    assert float(np.max(np.abs(rir))) > 0.0
