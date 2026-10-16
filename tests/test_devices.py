"""Tests for device band models (S4): ``acoustics.devices``.

Conventions verified here:
- response is exact at the preset frequency points, log-linear between them;
- link band is the intersection with both usable bands (no overlap → error);
- passive link gain ≤ 0 for the bundled presets;
- flat 0 dB devices reduce exactly to ``shannon_capacity``;
- device-aware capacity < flat Shannon, monotone in SNR;
- SPL headroom / limit hooks behave as documented.
"""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.acoustics.devices import LinkBand, Transducer, device_capacity
from auditory_handshake.config import DevicePreset, load_device_preset
from auditory_handshake.metrics.capacity import shannon_capacity

TX = Transducer(load_device_preset("small_loudspeaker"))
RX = Transducer(load_device_preset("mems_microphone"))


def _flat(kind: str) -> Transducer:
    """A 0 dB (no-loss) transducer with a wide usable band."""
    preset = DevicePreset(
        name=f"flat_{kind}",
        kind=kind,
        frequency_response_hz=(100.0, 1_000.0, 10_000.0, 20_000.0),
        frequency_response_db=(0.0, 0.0, 0.0, 0.0),
        usable_band_hz=(200.0, 16_000.0),
        max_spl_db=120.0,
    )
    return Transducer(preset)


# --- Transducer ---------------------------------------------------------------


def test_response_exact_at_preset_points() -> None:
    for t in (TX, RX):
        for f, db in zip(
            t.preset.frequency_response_hz, t.preset.frequency_response_db, strict=True
        ):
            assert t.response_db(f) == pytest.approx(db, abs=1e-9)


def test_response_log_linear_between_points() -> None:
    # 1 kHz (0 dB) and 2 kHz (-1 dB) on both presets: log-midpoint = -0.5 dB.
    for t in (TX, RX):
        mid = (1_000.0 * 2_000.0) ** 0.5  # geometric midpoint ≈ 1414 Hz
        lo, hi = t.response_db(1_000.0), t.response_db(2_000.0)
        assert lo > t.response_db(mid) > hi
        assert t.response_db(mid) == pytest.approx(-0.5, abs=1e-9)


def test_response_rejects_nonpositive_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        TX.response_db(0.0)
    with pytest.raises(ValueError, match="positive"):
        TX.response_db_array(np.array([1_000.0, 0.0]))


def test_speaker_rolls_off_below_500_hz() -> None:
    # Steep low-freq rolloff (PLAN §2.1): response strictly rises to 0 dB by 500 Hz.
    assert TX.response_db(100.0) < TX.response_db(200.0)
    assert TX.response_db(100.0) < TX.response_db(300.0) < 0.0
    assert TX.response_db(200.0) < TX.response_db(400.0) <= TX.response_db(500.0)


def test_mic_rolls_off_above_12_khz() -> None:
    vals = [RX.response_db(f) for f in (12_000.0, 14_000.0, 16_000.0, 18_000.0, 20_000.0)]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))


def test_spl_headroom_and_limit() -> None:
    assert TX.headroom_db(95.0) == pytest.approx(10.0)
    assert TX.over_spl_limit(105.0) is False
    assert TX.over_spl_limit(110.0) is True
    assert TX.headroom_db(110.0) < 0.0
    assert RX.headroom_db(120.0) == pytest.approx(0.0)


def test_transducer_exposes_preset() -> None:
    assert TX.name == "small_loudspeaker"
    assert TX.kind == "loudspeaker"
    assert TX.usable_band_hz == (300.0, 12_000.0)
    assert RX.in_usable_band(16_000.0) and not RX.in_usable_band(16_001.0)


# --- LinkBand ------------------------------------------------------------------


def test_band_intersection_is_device_limited() -> None:
    link = LinkBand.from_request(TX, RX, (200.0, 16_000.0))
    # speaker stops at 12 kHz; mic usable from 200 Hz → band is speaker-limited.
    assert link.band_hz == (300.0, 12_000.0)
    assert link.requested_hz == (200.0, 16_000.0)


def test_band_no_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        LinkBand.from_request(TX, RX, (12_500.0, 20_000.0))


def test_band_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="f1 < f2"):
        LinkBand.from_request(TX, RX, (5_000.0, 1_000.0))


def test_gain_is_passive_for_bundled_presets() -> None:
    link = LinkBand.from_request(TX, RX, (300.0, 16_000.0))
    for f in (400.0, 1_000.0, 5_000.0, 8_000.0, 11_000.0):
        assert link.gain_db(f) <= 1e-9
    assert link.average_gain_db() <= 1e-9


def test_gain_is_zero_at_the_flat_reference() -> None:
    # Both presets are 0 dB at 1 kHz → lossless at the reference.
    link = LinkBand.from_request(TX, RX, (300.0, 16_000.0))
    assert link.gain_db(1_000.0) == pytest.approx(0.0, abs=1e-9)


def test_gain_lower_at_band_edges() -> None:
    link = LinkBand.from_request(TX, RX, (300.0, 16_000.0))
    assert link.gain_db(400.0) < link.gain_db(1_000.0)
    assert link.gain_db(10_000.0) < link.gain_db(5_000.0)


# --- capacity ------------------------------------------------------------------


def test_flat_devices_reduce_to_shannon() -> None:
    f1, f2 = 300.0, 16_000.0
    cap = device_capacity(_flat("loudspeaker"), _flat("microphone"), (f1, f2), 30.0)
    assert cap == pytest.approx(shannon_capacity(f2 - f1, 30.0), rel=1e-3)


def test_device_capacity_below_flat_shannon() -> None:
    band = (300.0, 16_000.0)
    flat = shannon_capacity(band[1] - 300.0, 20.0)  # device band is 300–12 k
    dev = device_capacity(TX, RX, band, 20.0)
    assert 0.0 < dev < flat


def test_capacity_monotone_in_snr() -> None:
    band = (300.0, 16_000.0)
    assert (
        device_capacity(TX, RX, band, 10.0)
        < device_capacity(TX, RX, band, 20.0)
        < device_capacity(TX, RX, band, 30.0)
    )


def test_capacity_grows_with_bandwidth() -> None:
    assert device_capacity(TX, RX, (300.0, 1_000.0), 20.0) < device_capacity(
        TX, RX, (300.0, 16_000.0), 20.0
    )


def test_capacity_vanishes_at_low_snr() -> None:
    cap = device_capacity(TX, RX, (300.0, 16_000.0), -180.0)
    assert cap < 1.0  # bit/s: numerically 0 (SNR underflows to 0 in float64)


def test_capacity_converges_with_grid() -> None:
    band = (300.0, 16_000.0)
    coarse = device_capacity(TX, RX, band, 20.0, n_points=201)
    fine = device_capacity(TX, RX, band, 20.0, n_points=2_001)
    assert fine == pytest.approx(coarse, rel=5e-3)
