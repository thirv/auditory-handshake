"""S12.1 — Moving-source Doppler for the drone-arrival fast path.

PLAN §2.3: "Doppler ``f_d = f·v/c``: at 10 kHz and 5 m/s → ``f_d ≈ 146 Hz``,
which is enormous against a 20 Hz subcarrier spacing. At 1 kHz → 14.6 Hz. A
moving drone is a fast-fading, Doppler-heavy channel; pilots must be short and
re-estimated often."

This module renders the *Doppler component* of a moving source for a
transmitted waveform: the source signal is resampled in fixed-duration blocks,
each block at the Doppler rate implied by the source's speed *during that
block* (:func:`moving_source`). A constant speed is the S12.1 baseline (the
``drone_arrival`` scenario's ``drone_speed_ms``); a per-block speed profile
supports the variable-speed cases that land in S12.2.

Physics (PLAN §2.3): a source moving at closing speed ``v`` toward a
stationary receiver produces ``f_obs = f·c/(c − v)`` and
``f_d = f·v/(c − v)`` (receding is ``v < 0``). The resampling factor per unit
time is therefore ``c/(c − v)`` applied to the source-walk position.

Conventions (aligned with :mod:`~auditory_handshake.acoustics.noise` and
:mod:`~auditory_handshake.acoustics.channel`):

- Output is a 1-D ``float64`` pressure waveform of ``round(duration * fs)``
  samples.
- The model is *deterministic*: the same inputs reproduce bit-identical
  waveforms (no RNG in the chain).
- All engine dependence stays in :mod:`auditory_handshake.acoustics` so
  simulation code never reaches into the resampling primitive directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "SPEED_OF_SOUND_M_S",
    "doppler_factor",
    "measure_doppler_hz",
    "moving_source",
    "trajectory",
]

#: Speed of sound in air at 20 °C (m/s). The standard room-temperature default;
#: all functions accept a different ``speed_of_sound_m_s`` when needed.
SPEED_OF_SOUND_M_S: float = 343.0

#: Upper bound on the number of resampling blocks (a sanity guard against a
#: pathological ``block_duration_seconds`` producing millions of tiny blocks).
_MAX_BLOCKS = 10_000


def _validate_fs_duration(fs: float, duration_seconds: float) -> int:
    """Validate ``fs``/``duration`` and return the output sample count ``n``."""
    if not (np.isfinite(fs) and fs > 0.0):
        raise ValueError(f"fs must be a positive finite number, got {fs!r}")
    if not (np.isfinite(duration_seconds) and duration_seconds > 0.0):
        raise ValueError(
            f"duration_seconds must be a positive finite number, got {duration_seconds!r}"
        )
    return int(round(duration_seconds * fs))


def doppler_factor(speed_ms: float, speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S) -> float:
    """Resampling (frequency) factor for a source moving at ``speed_ms``.

    ``speed_ms`` is the closing speed toward a stationary receiver: positive
    (approaching) raises the received frequency, negative (receding) lowers it
    (``f_obs = f·c/(c − v)``; PLAN §2.3).

    Raises:
        ValueError: If ``speed_ms`` is not a finite speed strictly below the
            speed of sound, or ``speed_of_sound_m_s`` is not positive.
    """
    if not (np.isfinite(speed_of_sound_m_s) and speed_of_sound_m_s > 0.0):
        raise ValueError(
            f"speed_of_sound_m_s must be a positive finite number, got {speed_of_sound_m_s!r}"
        )
    if not (np.isfinite(speed_ms) and abs(speed_ms) < speed_of_sound_m_s):
        raise ValueError(
            f"speed_ms must be a finite value with |v| < c ({speed_of_sound_m_s} m/s), "
            f"got {speed_ms!r}"
        )
    return speed_of_sound_m_s / (speed_of_sound_m_s - speed_ms)


def _validate_speeds(
    speed_ms: float | Sequence[float],
    n_blocks: int,
    speed_of_sound_m_s: float,
) -> np.ndarray:
    """Normalise ``speed_ms`` (scalar or per-block) to an ``n_blocks`` array."""
    if isinstance(speed_ms, (int, float)):
        profile = np.full(n_blocks, float(speed_ms), dtype=np.float64)
    else:
        profile = np.asarray(speed_ms, dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError(f"speed_ms must be a scalar or a 1-D sequence, got shape {profile.shape}")
    if profile.size != n_blocks:
        raise ValueError(
            f"per-block speed_ms must have exactly {n_blocks} entries (one per block), "
            f"got {profile.size}"
        )
    if not np.all(np.isfinite(profile)) or np.any(np.abs(profile) >= speed_of_sound_m_s):
        raise ValueError(
            "every speed_ms entry must be finite with |v| < "
            f"{speed_of_sound_m_s} m/s, got {profile.tolist()}"
        )
    return profile


def trajectory(
    speed_ms: float | Sequence[float],
    duration_seconds: float,
    initial_distance_m: float,
    fs: float = 16000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Source ``position_m`` and ``distance_m`` vs. time along a 1-D path.

    The source starts at ``initial_distance_m`` in front of the receiver
    (position > 0) and moves along the line of sight: positive ``speed_ms``
    (or entries of a per-block profile) closes on the receiver, negative
    recedes. The path is piecewise-linear per sample interval, matching the
    piecewise-constant Doppler model in :func:`moving_source`.

    Returns:
        ``(position_m, distance_m)`` — 1-D ``float64`` arrays of
        ``round(duration_seconds * fs)`` samples; ``distance_m`` is
        ``abs(position_m)`` (the receiver is the origin).

    Raises:
        ValueError: If the geometry is invalid (see
            :func:`_validate_fs_duration` and :func:`doppler_factor`).
    """
    n = _validate_fs_duration(fs, duration_seconds)
    if not (np.isfinite(initial_distance_m) and initial_distance_m > 0.0):
        raise ValueError(
            f"initial_distance_m must be a positive finite number, got {initial_distance_m!r}"
        )

    profile = _validate_speeds(speed_ms, n, SPEED_OF_SOUND_M_S)
    # Per-sample displacement (m) at sample i = v · (i/fs): sample i sits at
    # t = i/fs, so the source starts exactly at ``initial_distance_m``
    # (position[0] == initial) and covers v·i/fs by sample i.
    displacement = (profile * np.arange(n)) / fs
    position = initial_distance_m - displacement
    return position.astype(np.float64), np.abs(position).astype(np.float64)


def moving_source(
    signal: np.ndarray,
    fs: float,
    speed_ms: float | Sequence[float],
    duration_seconds: float,
    block_duration_seconds: float,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> np.ndarray:
    """Render ``duration_seconds`` of ``signal`` heard from a moving source.

    The output is divided into contiguous blocks of
    ``round(block_duration_seconds * fs)`` samples (the last block is shorter
    when ``duration_seconds`` is not a multiple of the block length). Within
    each block the source is treated as moving at the constant speed assigned
    to that block (``speed_ms`` scalar = same speed every block — the S12.1
    baseline; a 1-D ``speed_ms`` sequence = one closing speed per block, in
    time order), so the source-walk advances at the Doppler factor
    :func:`doppler_factor`. Walk positions are accumulated across blocks, so
    the received waveform is phase-continuous at block boundaries.

    Args:
        signal: 1-D source waveform (pressure samples). Must be at least as
            long as the total source-walk required by the motion profile
            (walk > duration when approaching); unused tail is ignored.
        fs: Sampling rate (Hz) of both ``signal`` and the output.
        speed_ms: Closing speed (m/s, + = approaching) — scalar or one value
            per output block.
        duration_seconds: Length of the returned waveform in seconds.
        block_duration_seconds: Resampling block length in seconds; must be in
            ``(0, duration_seconds]`` and yield at most 10_000 blocks.
        speed_of_sound_m_s: Speed of sound (m/s); default 20 °C air.

    Returns:
        A 1-D ``float64`` waveform of ``round(duration_seconds * fs)`` samples.

    Raises:
        ValueError: If the geometry, the speed profile, or the ``signal``
            length is invalid (see the helpers above).
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 1 or sig.size < 2:
        raise ValueError(f"signal must be a 1-D array with at least 2 samples, got {sig.shape}")
    if not np.all(np.isfinite(sig)):
        raise ValueError("signal must contain only finite values")

    n = _validate_fs_duration(fs, duration_seconds)
    if not (
        np.isfinite(block_duration_seconds)
        and block_duration_seconds > 0.0
        and block_duration_seconds <= duration_seconds
    ):
        raise ValueError(
            f"block_duration_seconds must be in (0, {duration_seconds}] s, "
            f"got {block_duration_seconds!r}"
        )
    block_base = int(round(block_duration_seconds * fs))
    n_blocks = n // block_base
    if n_blocks < 1 or n_blocks > _MAX_BLOCKS:
        raise ValueError(
            f"block_duration_seconds must yield 1..{_MAX_BLOCKS} blocks for {n} samples, "
            f"got {n_blocks}"
        )

    profile = _validate_speeds(speed_ms, n_blocks, speed_of_sound_m_s)
    factors = np.array([doppler_factor(v, speed_of_sound_m_s) for v in profile])
    block_lengths = np.full(n_blocks, block_base, dtype=np.int64)
    block_lengths[-1] = n - (n_blocks - 1) * block_base

    # Accumulated source-walk position (in source samples) at each block edge:
    # block k walks ``factor_k * len_k`` source samples over its ``len_k``
    # output samples, starting where block k−1 ended (phase continuity).
    walk_edges = np.concatenate(
        [[0.0], np.cumsum(np.array([f * L for f, L in zip(factors, block_lengths, strict=True)]))]
    )
    if walk_edges[-1] > sig.size:
        raise ValueError(
            f"the motion profile requires {walk_edges[-1]:.0f} source samples "
            "(source-walk beyond the end of the approach), but signal has only "
            f"{sig.size}; pass a longer source signal"
        )

    out = np.empty(n, dtype=np.float64)
    cursor = 0
    for k, raw_length in enumerate(block_lengths):
        length = int(raw_length)
        start = walk_edges[k]
        end = walk_edges[k + 1]
        positions = np.linspace(start, end, length, endpoint=False)
        out[cursor : cursor + length] = np.interp(positions, np.arange(sig.size), sig)
        cursor += length

    return out


def measure_doppler_hz(
    received: np.ndarray,
    fs: float,
    nominal_hz: float,
) -> float:
    """Estimate the observed frequency shift of a steady received tone.

    A zero-crossing estimate of the received frequency (robust and bin-free,
    unlike a short-DFT estimate) minus ``nominal_hz``; the sign tells
    approach (positive) vs. recession (negative). Intended for steady
    (constant-speed) segments — the S12.1 regime — and for verifying
    :func:`moving_source` against the ``f·c/(c − v)`` prediction.

    Raises:
        ValueError: If the window is too short (needs ``≥ 25`` samples and at
            least one zero crossing), ``nominal_hz`` is not positive, or
            ``received`` is not finite.
    """
    x = np.asarray(received, dtype=np.float64)
    if x.ndim != 1 or x.size < 25:
        raise ValueError("received must be a 1-D array with at least 25 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("received must contain only finite values")
    if not (np.isfinite(nominal_hz) and nominal_hz > 0.0):
        raise ValueError(f"nominal_hz must be a positive finite number, got {nominal_hz!r}")

    signs = np.sign(x)
    signs = signs[signs != 0]
    if signs.size < 2:
        raise ValueError("received must have at least one zero crossing (no sign change)")
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    if crossings < 1:
        raise ValueError("received must have at least one zero crossing (no sign change)")

    measured_hz = crossings * fs / (2.0 * (x.size - 1))
    return float(measured_hz - nominal_hz)
