"""Room acoustics: theoretical reverberation time (RT60) of a room.

Computes the Sabine or Eyring reverberation time for the room described by a
:class:`~auditory_handshake.config.RoomPreset`.

The preset is modelled as a pyroomacoustics ``ShoeBox`` whose six walls are built
from the per-face absorption coefficients (``x`` -> west/east, ``y`` -> south/north,
``z`` -> floor/ceiling) over the preset's ``dimensions_m``. The single-band RT60 is
the bandwidth-weighted harmonic mean of the per-band closed-form values computed by
pyroomacoustics (``ShoeBox.rt60_theory``), the standard way to collapse the
frequency-dependent result into one number.

.. note::
    The project's standard RT60 value is the one computed for a 48 kHz sample rate
    (see the ``sample_rate_hz`` constant in :mod:`auditory_handshake.config`);
    :func:`calculate_rt60` defaults to that rate so multi-band presets are resampled
    onto the same octave-band grid.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pyroomacoustics as pra

if TYPE_CHECKING:
    from auditory_handshake.config import RoomPreset

__all__ = ["calculate_rt60"]

# pyroomacoustics wall names, keyed by the room-preset axis they belong to.
# The ShoeBox axes map onto the walls as: x -> west/east, y -> south/north,
# z -> floor/ceiling (see ``pyroomacoustics.room.ShoeBox``).
_AXIS_TO_WALLS: dict[str, tuple[str, str]] = {
    "x": ("west", "east"),
    "y": ("south", "north"),
    "z": ("floor", "ceiling"),
}

# Closed-form equations supported by pyroomacoustics.
_SUPPORTED_FORMULAS = ("sabine", "eyring")


def _build_material(values: Sequence[float], frequencies_hz: Sequence[float]) -> pra.Material:
    """Create a pyroomacoustics :class:`~pyroomacoustics.parameters.Material`.

    A single absorption coefficient yields a frequency-flat material. Multiple
    coefficients are paired with the preset's band-centre frequencies to form a
    multi-band material (pyroomacoustics resamples it onto the room's octave bands).
    """
    coeffs = [float(v) for v in values]
    if len(coeffs) == 1:
        return pra.Material(coeffs[0])
    if len(coeffs) != len(frequencies_hz):
        raise ValueError(
            "Number of absorption coefficients "
            f"({len(coeffs)}) does not match the number of band-centre "
            f"frequencies ({len(frequencies_hz)})."
        )
    return pra.Material({"coeffs": coeffs, "center_freqs": [float(f) for f in frequencies_hz]})


def _materials_for_preset(room_preset: RoomPreset) -> dict[str, pra.Material]:
    """Map a preset's per-face coefficients onto the six ``ShoeBox`` walls."""
    materials: dict[str, pra.Material] = {}
    for axis, (wall_a, wall_b) in _AXIS_TO_WALLS.items():
        material = _build_material(room_preset.absorption[axis], room_preset.frequencies_hz)
        materials[wall_a] = material
        materials[wall_b] = material
    return materials


def calculate_rt60(
    room_preset: RoomPreset,
    formula: str = "sabine",
    fs: float = 48000.0,
) -> float:
    """Compute the theoretical reverberation time (RT60) of a room, in seconds.

    The room is modelled as a pyroomacoustics ``ShoeBox`` built from
    ``room_preset.dimensions_m`` (metres) and the per-face absorption coefficients in
    ``room_preset.absorption``. The per-band values follow the chosen closed-form
    equation::

        Sabine: RT60(f) = (24 ln(10) / c) * V / (A(f) + 4 m V)
        Eyring: RT60(f) = (24 ln(10) / c) * V / (S ln(1 - alpha(f)) + 4 m V)

    where ``V`` is the room volume, ``S`` the total surface, ``A(f)`` the total
    absorption and ``alpha(f)`` the average absorption in band ``f``, ``m`` the air
    absorption, and ``c`` the speed of sound (343 m/s). These are combined into a
    single value via a bandwidth-weighted harmonic mean (pyroomacoustics'
    ``rt60_theory``).

    Parameters
    ----------
    room_preset:
        Room description (``dimensions_m`` and per-face absorption coefficients).
    formula:
        Closed-form equation to use -- ``'sabine'`` or ``'eyring'`` (case-insensitive).
        Default is ``'sabine'``.
    fs:
        Sampling rate (Hz). It fixes the octave-band grid onto which multi-band
        absorption coefficients are resampled before the harmonic-mean combination;
        for flat (single-coefficient) materials it has no effect. Default is 48000,
        the project's standard sample rate.

    Returns
    -------
    float
        The RT60 in seconds. Returns ``inf`` (and emits a :class:`RuntimeWarning`)
        when the room has no total absorption, in which case the sound field never
        decays and the reverberation time is unbounded.

    Raises
    ------
    ValueError
        If ``formula`` is not a supported equation, or a face's number of absorption
        coefficients does not match the number of band-centre frequencies.
    """
    key = formula.lower()
    if key not in _SUPPORTED_FORMULAS:
        supported = ", ".join(_SUPPORTED_FORMULAS)
        raise ValueError(f"Unsupported RT60 formula {formula!r}; expected one of: {supported}.")

    coefficients = np.concatenate(
        [np.asarray(room_preset.absorption[axis], dtype=float) for axis in _AXIS_TO_WALLS]
    )
    if coefficients.size == 0 or np.allclose(coefficients, 0.0):
        warnings.warn(
            "The room has no total absorption, so the theoretical RT60 is infinite "
            "(a perfectly reflecting room never decays).",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf")

    room = pra.ShoeBox(
        list(room_preset.dimensions_m),
        fs=fs,
        materials=_materials_for_preset(room_preset),
    )

    # Zero absorption in individual bands makes pyroomacoustics emit divide-by-zero
    # RuntimeWarnings while still producing a finite harmonic mean; keep those quiet
    # and surface a single, explicit warning only if the overall result is unbounded.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rt60 = room.rt60_theory(formula=key)

    if np.isinf(rt60):
        warnings.warn(
            "The room's absorption is too low for a finite RT60; returning inf.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf")

    return float(rt60)
