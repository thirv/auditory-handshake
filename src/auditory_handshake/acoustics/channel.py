"""Deterministic per-channel RIR generation.

Given a :class:`~auditory_handshake.config.RoomPreset` and a target RT60
(seconds), produce a mono impulse response for *one* channel.

The RIR is synthesised with :mod:`pyroomacoustics` (image-source Shoebox model).
The preset defines the room *geometry* and a *base* absorption pattern; the
target RT60 is achieved by uniformly scaling the base Sabine absorption
coefficients so the theoretical RT60 matches the requested value. A
frequency-independent (flat) RT60 is what the baseline channel model needs.

The pipeline (PLAN.md §2 / §5) anchors the direct-sound onset at sample 0,
keeps a reverberant tail (a 0.25 s floor, extended to 3x the target RT60 for
reverberant rooms), and normalises to unit peak. The result
is deterministic for a fixed ``preset``/``rt60_seconds``/``fs`` and independent
of any global random seed (no stochastic step in the chain).

On top of :func:`compute_rir`, :func:`analyze_channel` binds the room preset,
the generated RIR, and the S10.1 channel metrics (Schroeder RT60, ``tau_mean``,
``tau_rms``, ``B50``/``B90``) into one frozen :class:`ChannelAnalysis` object —
the analysis unit the S11 channel characterisation sim consumes when it emits
``configs/phy_constraints.yaml``.

For an RT60 sweep, :func:`compute_rir_sweep` (or its raw-matrix counterpart
:func:`sweep_rir_matrix`) runs :func:`compute_rir` once per target RT60 and
returns the per-target :class:`ChannelAnalysis` objects (or a single
``(N, L)`` matrix) with identical ``fs``/separation parameters, so the whole
sweep is the deterministic per-target result — the same RIR you'd get calling
``compute_rir`` for each target individually, just batched for convenience.

This module owns *channel* processing; :mod:`.room` owns room/preset
construction and :mod:`:.link_budget` owns the RF baseline, keeping the three
concerns decoupled.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyroomacoustics as pra

from auditory_handshake.config import RoomPreset
from auditory_handshake.metrics import channel_metrics

# The stored RIR keeps at least this much reverberant tail (PLAN.md §5 floor),
# so near-anechoic rooms still get a stable, non-empty response.
_MIN_TAIL_SECONDS = 0.25

# The tail is kept out to ~3x the target RT60: that captures the bulk of the
# exponential decay so the Schroeder / B50 / B90 / tau metrics are not
# contaminated by an arbitrary early cut, and reverberant rooms are no longer
# clipped before their decay becomes measurable.
_TAIL_RT60_MULTIPLIER = 3.0

# Maximum image-source reflection order. The zero-padding fallback in
# :func:`compute_rir` guards the rare case where the natural RIR is shorter than
# the requested (0.25 s floor / 3x-RT60) tail regardless of room size.
_MAX_ORDER = 40


def _tail_length_seconds(rt60_seconds: float) -> float:
    """Output tail length (s): the max of the 0.25 s floor and 3x the RT60.

    Keeps a stable minimum tail for near-anechoic rooms while extending to
    ``3 * rt60_seconds`` for reverberant rooms so the decay is fully captured.
    """
    return max(_MIN_TAIL_SECONDS, _TAIL_RT60_MULTIPLIER * max(0.0, float(rt60_seconds)))


# Source/microphone sit at the room centre a fixed, realistic distance apart
# along +x. That separation sets the direct-to-reverb ratio (DRR): the
# image-source model weights each path by 1/r, so a centimetre-scale gap makes
# the direct path dominate every reflection (DRR ~33-42 dB) and the decay-tail
# metrics (Schroeder RT60, B50/B90, tau) lock to the direct spike, independent
# of RT60. A ~0.3 m separation restores a plausible direct/reverb balance so
# those metrics track the target RT60; it also keeps the separation finite and
# non-degenerate (the image-source model rejects a zero-length one).
_SRC_MIC_OFFSET_M = 0.3

# Shoebox face -> (wall_a, wall_b), per :mod:`pyroomacoustics` naming.
_FACE_WALLS = {
    "x": ("west", "east"),
    "y": ("south", "north"),
    "z": ("floor", "ceiling"),
}


def _build_material(coeffs: Sequence[float], frequencies_hz: Sequence[float]) -> pra.Material:
    """Map a per-band Sabine coefficient array to a :class:`pra.Material`."""
    if len(coeffs) == 1:
        return pra.Material(float(coeffs[0]))
    return pra.Material(
        {"coeffs": [float(c) for c in coeffs], "center_freqs": [float(f) for f in frequencies_hz]}
    )


def _scaled_materials(preset: RoomPreset, scale: float) -> dict[str, pra.Material]:
    """Build the six Shoebox face materials from ``preset`` scaled by ``scale``.

    Like :func:`auditory_handshake.acoustics.room._materials_for_preset`, but each
    Sabine coefficient is multiplied by ``scale`` and clamped to ``[0, 1]`` so
    the scaled material stays a valid absorption coefficient.
    """
    out: dict[str, pra.Material] = {}
    for face, (wall_a, wall_b) in _FACE_WALLS.items():
        coeffs = [min(1.0, max(0.0, c * scale)) for c in preset.absorption[face]]
        material = _build_material(coeffs, preset.frequencies_hz)
        out[wall_a] = material
        out[wall_b] = material
    return out


def _absorption_scale(preset: RoomPreset, rt60_seconds: float) -> float:
    """Uniform scale on the base absorption that hits ``rt60_seconds``.

    From Sabine's law ``RT60 = 0.161 * V / A_total`` the target total absorption
    is ``A = 0.161 * V / RT60``; dividing by the preset's base absorption gives
    the uniform coefficient scale.

    A non-positive ``rt60_seconds`` models "no measurable reverb" -> full
    absorption on every face (near-pure direct impulse). A large value drives
    the scale toward zero (low absorption, long tail).
    """
    dx, dy, dz = preset.dimensions_m
    volume = float(dx) * float(dy) * float(dz)

    # Each face pair covers twice the opposite-face area.
    face_areas = {"x": 2.0 * dy * dz, "y": 2.0 * dx * dz, "z": 2.0 * dx * dy}
    base_absorption = sum(
        float(np.mean(preset.absorption[face])) * face_areas[face] for face in ("x", "y", "z")
    )

    if rt60_seconds <= 0.0 or base_absorption <= 0.0:
        return 1e4  # saturate every face at alpha = 1 -> direct sound only

    target_absorption = 0.161 * volume / float(rt60_seconds)
    return max(1e-3, target_absorption / base_absorption)


def compute_rir(
    room_preset: RoomPreset,
    rt60_seconds: float,
    *,
    fs: float = 16000.0,
    src_mic_offset_m: float = _SRC_MIC_OFFSET_M,
) -> np.ndarray:
    """Compute a deterministic mono RIR for a room preset and target RT60.

    Parameters
    ----------
    room_preset:
        The room geometry + base per-face absorption to scale from.
    rt60_seconds:
        Target reverberation time (seconds). The base absorption is uniformly
        scaled to reach this value. ``<= 0`` yields a near-pure direct impulse.
    fs:
        Sampling rate (Hz). The output keeps a tail of
        ``max(0.25 s, 3 * rt60_seconds)`` (see :func:`_tail_length_seconds`),
        so the sample count scales linearly with ``fs``.
    src_mic_offset_m:
        Separation (metres) between the source and microphone, both
        centred in the room. A larger value increases the reverb
        contribution relative to the direct path (lower DRR); a
        smaller value makes the direct path dominate (higher DRR).
        Must be finite and non-negative. Default: ``0.3 m``.

    Returns
    -------
    np.ndarray
        A 1-D ``float32`` array of length ``int(round(_tail_length_seconds(
        rt60_seconds) * fs))`` with the onset (first non-negligible sample)
        anchored at sample 0; the strongest sample is normalised to unit
        amplitude and sits at a non-zero index (the propagation delay), *not*
        sample 0 — align to it via ``argmax`` if timing is needed.
    """
    scale = _absorption_scale(room_preset, rt60_seconds)
    room = pra.ShoeBox(
        [float(v) for v in room_preset.dimensions_m],
        fs=int(fs),
        materials=_scaled_materials(room_preset, scale),
        max_order=_MAX_ORDER,
    )

    # Source/microphone sit at the room centre, separated by the fixed, realistic
    # _SRC_MIC_OFFSET_M above (see its rationale for the DRR argument). The direct
    # path is still the shortest, so the onset is well defined and deterministic.
    center = [float(v) / 2.0 for v in room_preset.dimensions_m]
    # Keep the microphone strictly inside the room along the offset axis; an
    # out-of-bounds or zero-separation mic would make the direct path and the
    # reflection geometry undefined.  No-op for the bundled presets (minimum
    # dimension >= 1.6 m with the 0.3 m default offset); guards custom presets.
    max_x_offset = max(center[0] - 1e-3, 1e-3)
    src_mic_offset_m = float(np.clip(src_mic_offset_m, -max_x_offset, max_x_offset))
    room.add_soundsource(pra.SoundSource(position=center))
    room.add_microphone([center[0] + src_mic_offset_m, center[1], center[2]])
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float64)

    # Anchor the first non-negligible sample (the onset) to index 0 — note the
    # direct path (the peak, normalised to unit amplitude) sits at a later,
    # non-zero index (the propagation delay), *not* index 0. Keep a tail long
    # enough to contain the decay (max of the 0.25 s floor and 3x the target
    # RT60), and normalise to unit peak.
    n_tail = int(round(_tail_length_seconds(rt60_seconds) * float(fs)))
    magnitude = np.abs(rir)
    significant = np.flatnonzero(magnitude > 1e-9)
    onset = int(significant[0]) if significant.size else 0
    tail = rir[onset : onset + n_tail]
    if tail.size < n_tail:
        tail = np.pad(tail, (0, n_tail - tail.size), mode="constant")
    peak = float(np.max(np.abs(tail)))
    if peak > 0.0:
        tail = tail / peak

    return tail.astype(np.float32)


def sweep_rir_matrix(
    room_preset: RoomPreset,
    rt60_seconds: Sequence[float],
    *,
    fs: float = 16000.0,
    src_mic_offset_m: float = _SRC_MIC_OFFSET_M,
) -> np.ndarray:
    """Compute a batch of mono RIRs for one room across an RT60 sweep.

    This is the raw-matrix counterpart of :func:`compute_rir_sweep`: it runs
    :func:`compute_rir` once per target RT60 (identical ``fs`` / separation)
    and stacks the results into a single ``(N, L)`` matrix, where ``N`` is the
    number of target RT60 values and ``L`` is the (constant) RIR length for
    that ``fs``. Each row ``i`` is exactly the RIR you would get calling
    ``compute_rir(room_preset, rt60_seconds[i], fs=fs,
    src_mic_offset_m=src_mic_offset_m)`` directly — deterministic, unit peak,
    onset anchored at index 0.

    Parameters
    ----------
    room_preset:
        The room geometry + base per-face absorption to scale from.
    rt60_seconds:
        A non-empty sequence of target reverberation times (seconds). Every
        value must be finite and non-negative (mirrors :func:`compute_rir`);
        an empty sequence raises ``ValueError``.
    fs:
        Sampling rate (Hz), shared by every row (passed to :func:`compute_rir`).
    src_mic_offset_m:
        Source–microphone separation (metres), shared by every row.

    Returns
    -------
    np.ndarray
        A 2-D ``float32`` array of shape ``(N, L)``; ``L`` is ``int(round(
        _tail_length_seconds(max(rt60_seconds)) * fs))`` (the longest tail),
        with each row left-padded to that width and normalised to unit peak.
    """
    rt60_list = [float(v) for v in rt60_seconds]
    if not rt60_list:
        raise ValueError("sweep_rir_matrix: rt60_seconds must be non-empty")
    for i, v in enumerate(rt60_list):
        if not np.isfinite(v):
            raise ValueError(f"sweep_rir_matrix: rt60[{i}] must be finite (got {v!r})")
        if v < 0.0:
            raise ValueError(f"sweep_rir_matrix: rt60[{i}] must be non-negative (got {v!r})")

    max_rt60 = max(rt60_list)
    n_tail = int(round(_tail_length_seconds(max_rt60) * float(fs)))
    rows = [
        np.asarray(
            compute_rir(room_preset, v, fs=fs, src_mic_offset_m=src_mic_offset_m), dtype=np.float32
        )
        for v in rt60_list
    ]
    matrix = np.zeros((len(rt60_list), n_tail), dtype=np.float32)
    for i, row in enumerate(rows):
        width = min(row.size, n_tail)
        matrix[i, :width] = row[:width]
    return matrix


def compute_rir_sweep(
    room_preset: RoomPreset,
    rt60_seconds: Sequence[float],
    *,
    fs: float = 16000.0,
    src_mic_offset_m: float = _SRC_MIC_OFFSET_M,
) -> list[ChannelAnalysis]:
    """Analyse one room across an RT60 sweep, one :class:`ChannelAnalysis` per target.

    Convenience over :func:`analyze_channel`: for every target RT60 in
    ``rt60_seconds`` (non-empty; each value finite and non-negative) it returns
    the same :class:`ChannelAnalysis` you'd get from a direct
    ``analyze_channel(room_preset, v, fs=fs, src_mic_offset_m=src_mic_offset_m)``
    call — same preset, same ``fs``, same RIR, same S10.1 channel metrics —
    just batched, so the whole sweep is as deterministic as the single-point
    result.

    Parameters
    ----------
    room_preset:
        The room geometry + base per-face absorption to scale from.
    rt60_seconds:
        A non-empty sequence of target reverberation times (seconds). Every
        value must be finite and non-negative; an empty sequence raises
        ``ValueError``.
    fs:
        Sampling rate (Hz), shared by every entry.
    src_mic_offset_m:
        Source–microphone separation (metres), shared by every entry.

    Returns
    -------
    list[ChannelAnalysis]
        One analysis per target RT60, in input order (same length as
        ``rt60_seconds``).
    """
    rt60_list = [float(v) for v in rt60_seconds]
    if not rt60_list:
        raise ValueError("compute_rir_sweep: rt60_seconds must be non-empty")
    for i, v in enumerate(rt60_list):
        if not np.isfinite(v):
            raise ValueError(f"compute_rir_sweep: rt60[{i}] must be finite (got {v!r})")
        if v < 0.0:
            raise ValueError(f"compute_rir_sweep: rt60[{i}] must be non-negative (got {v!r})")

    return [
        analyze_channel(room_preset, v, fs=fs, src_mic_offset_m=src_mic_offset_m) for v in rt60_list
    ]


def _finite_or_none(value: float) -> float | None:
    """A YAML/JSON-friendly float: ``inf``/``nan`` become ``null``."""
    return float(value) if np.isfinite(value) else None


@dataclass(frozen=True, eq=False)
class ChannelAnalysis:
    """One channel, fully characterised: preset + RIR + S10.1 metrics (S10.2).

    Binds the room preset, the deterministic mono RIR from
    :func:`compute_rir`, and the channel metrics (Schroeder RT60,
    ``tau_mean``, ``tau_rms``, ``B50``/``B90`` — all in seconds) into a single
    immutable analysis object, the unit the S11 channel-characterisation sim
    iterates over when it emits ``configs/phy_constraints.yaml``.

    ``eq=False``: object identity comparison (the ``rir`` ndarray would make
    generated ``==`` ambiguous); compare fields (or the dict from
    :meth:`to_dict`) explicitly.
    """

    preset: RoomPreset
    rt60_target_seconds: float
    fs: float
    rir: np.ndarray
    rt60_schroeder: float
    tau_mean: float
    tau_rms: float
    b50: float
    b90: float

    @property
    def tail_seconds(self) -> float:
        """Length of the stored RIR in seconds (see :func:`_tail_length_seconds`)."""
        return float(self.rir.size) / float(self.fs)

    def to_dict(self) -> dict[str, Any]:
        """Flat, serialisable snapshot (JSON/YAML safe) of this analysis.

        Contains the preset (via :meth:`RoomPreset.to_dict`), the target RT60,
        ``fs``, the RIR sample count, and the five metrics. The raw RIR array is
        not inlined (consumers regenerate it deterministically from
        ``preset``/``rt60_target_seconds``/``fs``). A non-finite metric (e.g. an
        anechoic Schroeder RT60 of ``inf``) is serialised as ``None``.
        """
        out: dict[str, Any] = {
            "preset": self.preset.to_dict(),
            "rt60_target_seconds": float(self.rt60_target_seconds),
            "fs": float(self.fs),
            "rir_samples": int(self.rir.size),
            "rt60_schroeder": _finite_or_none(self.rt60_schroeder),
            "tau_mean": float(self.tau_mean),
            "tau_rms": float(self.tau_rms),
            "b50": float(self.b50),
            "b90": float(self.b90),
        }
        return out


def analyze_channel(
    room_preset: RoomPreset,
    rt60_seconds: float,
    *,
    fs: float = 16000.0,
    src_mic_offset_m: float = _SRC_MIC_OFFSET_M,
) -> ChannelAnalysis:
    """Compute the RIR for a preset at a target RT60 and measure its metrics.

    Thin, deterministic binding of :func:`compute_rir` and the S10.1 metrics
    (:func:`~auditory_handshake.metrics.channel_metrics.rt60_schroeder`,
    ``tau_mean``, ``tau_rms``, ``b50``, ``b90``) into one
    :class:`ChannelAnalysis`. No new physics: all parameters and defaults are
    the same as for ``compute_rir``.

    Parameters
    ----------
    room_preset:
        Room geometry + base per-face absorption to scale from.
    rt60_seconds:
        Target reverberation time (seconds); ``<= 0`` yields a near-pure
        direct impulse.
    fs:
        Sampling rate (Hz), passed straight through to :func:`compute_rir` and
        the metric functions.
    src_mic_offset_m:
        Source–microphone separation (metres); see :func:`compute_rir`.

    Returns
    -------
    ChannelAnalysis
        The preset, the mono RIR (unit peak; the direct path sits at the
        channel's propagation-delay index — a non-zero sample, *not* sample 0 —
        with pre-echo before it; 0.25 s floor / 3x-RT60 tail), and the five
        channel metrics in seconds.
    """
    rir = compute_rir(room_preset, rt60_seconds, fs=fs, src_mic_offset_m=src_mic_offset_m)
    return ChannelAnalysis(
        preset=room_preset,
        rt60_target_seconds=float(rt60_seconds),
        fs=float(fs),
        rir=rir,
        rt60_schroeder=channel_metrics.rt60_schroeder(rir, fs),
        tau_mean=channel_metrics.tau_mean(rir, fs),
        tau_rms=channel_metrics.tau_rms(rir, fs),
        b50=channel_metrics.b50(rir, fs),
        b90=channel_metrics.b90(rir, fs),
    )


def echo_tail_guard_s(rir: np.ndarray, fs: float, *, threshold_db: float = -30.0) -> float:
    r"""Length (s) of the reverberant tail that must survive a round trip.

    The channel model (PLAN.md §2 / §5) keeps the RIR out to a 0.25 s floor /
    3x-RT60 tail so that a single round trip does not wipe out the
    reverberation the detector relies on. This is a *guard* on that: given a
    captured RIR, it measures how long the tail actually survives down to
    ``threshold_db`` (default ``-30 dB``) below peak energy, so a user can
    check that an observed RIR is long enough before the round trip.

    Uses the backward Schroeder energy decay curve (the same integral the
    :func:`~auditory_handshake.metrics.channel_metrics.rt60_schroeder` /
    ``b50`` / ``b90`` metrics are built on)::

        E[i] = sum_{j >= i} h[j]^2,   t[i] = i / fs,   E[0] = 1.

    For a single-pole exponential tail ``E(t) = E(0) * exp(-t / tau)``, the
    crossing of ``E(0) * 10**(threshold_db / 10)`` is
    ``tau * ln(10) * |threshold_db| / 20`` (``-60 dB`` => ``6.91 * tau``,
    matching the Schroeder RT60 convention). The result is a continuous
    time, found by linear interpolation between the two samples bracketing
    the crossing.

    Parameters
    ----------
    rir:
        Mono impulse response.
    fs:
        Sampling rate (Hz); must be ``> 0``.
    threshold_db:
        Decay depth (dB below peak energy) to measure to. Must be a finite
        value ``< 0``; ``>= 0`` (or non-finite) returns ``0.0``.

    Returns
    -------
    float
        Tail length in seconds. ``0.0`` for an empty RIR, a non-positive
        threshold, or a signal with no energy (no peak).

        If the tail never decays to ``threshold_db`` within the captured RIR
        (common when the RIR is shorter than the tail, e.g. a 0.25 s-floor
        room with a long reverb time), the result is capped at the RIR's
        total duration and a :class:`RuntimeWarning` is emitted — treat the
        value as a *lower bound* on the true tail, not the exact decay
        length.

    Raises
    ------
    ValueError
        If ``fs <= 0`` or ``threshold_db`` is non-finite (a finite but
        non-negative threshold is handled by the ``0.0`` return above).
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if not np.isfinite(threshold_db):
        raise ValueError(f"threshold_db must be finite, got {threshold_db}")

    h = np.asarray(rir)
    if h.size == 0 or float(np.max(h**2)) <= 0.0:
        return 0.0
    if threshold_db >= 0.0:
        return 0.0

    e = np.cumsum(h[::-1] ** 2)[::-1]
    target = e[0] * 10.0 ** (threshold_db / 10.0)
    idx = int(np.argmax(e <= target))
    if idx == 0:
        # Energy never reaches the threshold within the captured RIR: the
        # tail is at least as long as the RIR itself. Report the RIR
        # duration as a lower bound and warn the caller.
        duration = float(h.size) / float(fs)
        warnings.warn(
            "echo_tail_guard_s: RIR tail does not decay to "
            f"{threshold_db:.1f} dB within {h.size} samples ({duration:.3f} s); "
            f"returning the RIR duration as a lower bound on the tail length",
            RuntimeWarning,
            stacklevel=2,
        )
        return duration

    t0 = (idx - 1) / fs
    t1 = idx / fs
    e0 = float(e[idx - 1])
    e1 = float(e[idx])
    frac = (e0 - target) / (e0 - e1) if e0 != e1 else 0.0
    return float(t0 + frac * (t1 - t0))
