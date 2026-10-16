"""Tests for :mod:`auditory_handshake.acoustics.room` (theoretical RT60)."""

import math

import numpy as np
import pyroomacoustics as pra
import pytest

from auditory_handshake.acoustics import calculate_rt60
from auditory_handshake.config import RoomPreset, list_room_presets, load_room_preset


def _flat(alpha: float, dims=(5.0, 4.0, 3.0)) -> RoomPreset:
    """A room with the same frequency-flat absorption on every face."""
    return RoomPreset(
        name="test",
        dimensions_m=tuple(dims),
        frequencies_hz=(1000.0,),
        absorption={"x": (alpha,), "y": (alpha,), "z": (alpha,)},
    )


def test_flat_sabine_is_finite_and_close_to_no_air_value():
    # No-air Sabine for a 5x4x3 room with 0.1 absorption: 0.161 * 60 / 9.4 ~ 1.028 s.
    rt60 = calculate_rt60(_flat(0.1), formula="sabine")
    assert math.isfinite(rt60)
    expected_no_air = 0.161 * 60.0 / (94.0 * 0.1)  # no-air Sabine ~ 1.028 s
    assert rt60 == pytest.approx(expected_no_air, rel=0.05)
    assert 0.9 < rt60 < 1.1


def test_sabine_is_greater_than_eyring():
    preset = _flat(0.1)
    sabine = calculate_rt60(preset, formula="sabine")
    eyring = calculate_rt60(preset, formula="eyring")
    assert 0.0 < eyring < sabine


def test_formula_is_case_insensitive():
    assert calculate_rt60(_flat(0.1), formula="Sabine") == pytest.approx(
        calculate_rt60(_flat(0.1), formula="sabine")
    )


def test_higher_absorption_gives_lower_rt60():
    assert calculate_rt60(_flat(0.3)) < calculate_rt60(_flat(0.1))


def test_larger_room_gives_longer_rt60():
    small = calculate_rt60(_flat(0.1, dims=(5.0, 4.0, 3.0)))
    large = calculate_rt60(_flat(0.1, dims=(10.0, 8.0, 6.0)))
    assert large > small


def test_flat_rt60_is_independent_of_sample_rate():
    # A flat material is not resampled, so the result must not depend on ``fs``.
    preset = _flat(0.1)
    assert calculate_rt60(preset, fs=48000) == pytest.approx(calculate_rt60(preset, fs=16000))


def test_multiband_flat_returns_finite_value():
    coeffs = (0.05, 0.06, 0.07, 0.08, 0.09)
    preset = RoomPreset(
        name="test",
        dimensions_m=(5.0, 4.0, 3.0),
        frequencies_hz=(125.0, 250.0, 500.0, 1000.0, 2000.0),
        absorption={"x": coeffs, "y": coeffs, "z": coeffs},
    )
    rt60 = calculate_rt60(preset)
    assert math.isfinite(rt60) and rt60 > 0.0


def test_zero_absorption_returns_inf_and_warns():
    with pytest.warns(RuntimeWarning):
        result = calculate_rt60(_flat(0.0))
    assert math.isinf(result) and result > 0.0


def test_invalid_formula_raises():
    with pytest.raises(ValueError, match="Unsupported RT60 formula"):
        calculate_rt60(_flat(0.1), formula="not-a-formula")


def test_coeff_count_mismatch_raises():
    preset = RoomPreset(
        name="test",
        dimensions_m=(5.0, 4.0, 3.0),
        frequencies_hz=(125.0, 250.0, 500.0),  # three bands, but two coefficients
        absorption={
            "x": (0.05, 0.06),
            "y": (0.05, 0.06),
            "z": (0.05, 0.06),
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        calculate_rt60(preset)


# ---------------------------------------------------------------------------
# Independent cross-validation of the RT60 algorithm
#
# These tests re-derive the reverberation time from the textbook Sabine/Eyring
# closed forms -- without calling :func:`calculate_rt60`'s own helpers or
# pyroomacoustics' ``rt60_theory`` / ``rt60_sabine`` / ``rt60_eyring`` -- and
# compare against the implementation. They guard against a regression in
# ``calculate_rt60`` *or* in the upstream combination it delegates to.
# ---------------------------------------------------------------------------

_FS = 48000.0  # project standard sample rate (calculate_rt60's default)
_C = 343.0  # speed of sound (pyroomacoustics default)
_K = 24.0 * np.log(10.0) / _C  # 24 ln(10) / c
_AXIS_TO_WALLS = {"x": ("west", "east"), "y": ("south", "north"), "z": ("floor", "ceiling")}


def _independent_shoebox(preset: RoomPreset) -> pra.ShoeBox:
    """Build the ``ShoeBox`` directly from the preset, in the test itself.

    Deliberately independent of :func:`auditory_handshake.acoustics.room`'s own
    material-assembly helpers, so the RT60 value is checked against a separate
    construction of the same room.
    """
    materials: dict[str, pra.Material] = {}
    for axis, (wall_a, wall_b) in _AXIS_TO_WALLS.items():
        coeffs = [float(v) for v in preset.absorption[axis]]
        if len(coeffs) == 1:
            material = pra.Material(coeffs[0])
        else:
            material = pra.Material(
                {"coeffs": coeffs, "center_freqs": [float(f) for f in preset.frequencies_hz]}
            )
        materials[wall_a] = material
        materials[wall_b] = material
    return pra.ShoeBox(list(preset.dimensions_m), fs=_FS, materials=materials)


def _band_coeff(wall: pra.Wall, band: int) -> float:
    """Per-band Sabine coefficient for ``wall`` (scalar for flat rooms)."""
    arr = np.asarray(wall.absorption, dtype=float)
    return float(arr[band]) if arr.ndim else float(arr)


def _rt60_textbook(room: pra.ShoeBox, formula: str) -> float:
    """Re-derive RT60 from the Sabine/Eyring closed forms.

    Each (band, wall) cell contributes
        Sabine: ``K*V / (S_w*alpha_w + 4*m*V)``
        Eyring: ``-K*V / (S_w*ln(1 - alpha_w) + 4*m*V)``
    and the result is the bandwidth-weighted harmonic mean
    ``sum(bw) / sum(bw / rt60_cell)`` -- the same structure as ``rt60_theory``
    but written from first principles here.
    """
    if room.is_multi_band:
        bandwidths = np.asarray(room.octave_bands.get_bw(), dtype=float)
    else:
        bandwidths = np.array([1.0])
    V = float(room.volume)
    inv_rt60 = 0.0
    for i, bw in enumerate(bandwidths):
        m = 0.0 if room.air_absorption is None else float(room.air_absorption[i])
        for wall in room.walls:
            area = float(wall.area())
            alpha = _band_coeff(wall, i)
            if formula == "sabine":
                cell = _K * V / (area * alpha + 4.0 * m * V)
            else:
                cell = -_K * V / (area * np.log(1.0 - alpha) + 4.0 * m * V)
            inv_rt60 += bw / cell
    return float(np.sum(bandwidths) / inv_rt60)


@pytest.mark.parametrize("formula", ["sabine", "eyring"])
@pytest.mark.parametrize("name", list_room_presets())
def test_rt60_matches_independently_built_shoebox(name: str, formula: str) -> None:
    """``calculate_rt60`` equals a directly-constructed ``ShoeBox.rt60_theory``."""
    preset = load_room_preset(name)
    # Tight tolerance (rel=1e-12): this delegates to the *same* pra.ShoeBox.rt60_theory
    # on a hand-built instance, so the arithmetic is bit-identical. The closed-form
    # test below needs a looser tolerance -- see why there.
    assert calculate_rt60(preset, formula=formula) == pytest.approx(
        _independent_shoebox(preset).rt60_theory(formula), rel=1e-12
    )


@pytest.mark.parametrize("formula", ["sabine", "eyring"])
@pytest.mark.parametrize("name", list_room_presets())
def test_rt60_matches_textbook_closed_form(name: str, formula: str) -> None:
    """``calculate_rt60`` equals the Sabine/Eyring closed form derived by hand."""
    room = _independent_shoebox(load_room_preset(name))
    # Loose tolerance on purpose (rel=1e-6, NOT 1e-12): this re-derives the formula
    # from first principles, so the octave-band harmonic-mean collapse accumulates in
    # a different summation order than pyroomacoustics' internal reduction, leaving
    # genuine float rounding differences. Do not "tighten" this back to 1e-12.
    assert calculate_rt60(load_room_preset(name), formula=formula) == pytest.approx(
        _rt60_textbook(room, formula), rel=1e-6
    )


def test_flat_room_matches_textbook_closed_form() -> None:
    """A single-band, uniform room reduces to the Sabine/Eyring formula itself."""
    dims = (5.0, 4.0, 3.0)
    alpha = 0.1
    preset = RoomPreset(
        name="flat",
        dimensions_m=dims,
        frequencies_hz=(1000.0,),
        absorption={"x": (alpha,), "y": (alpha,), "z": (alpha,)},
    )
    V = dims[0] * dims[1] * dims[2]
    S = 2.0 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])
    sabine = _K * V / (S * alpha)
    eyring = -_K * V / (S * np.log(1.0 - alpha))
    assert calculate_rt60(preset, formula="sabine") == pytest.approx(sabine, rel=1e-6)
    assert calculate_rt60(preset, formula="eyring") == pytest.approx(eyring, rel=1e-6)
