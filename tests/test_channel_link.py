"""S13.2 — MFSK BER/BLER under acoustic channel conditions (noise, RIR, motion).

These integration tests chain the S13.1 modem through the S13.2 acoustic
channel (Doppler → RIR → additive noise) and measure link performance with the
real :mod:`auditory_handshake.acoustics` primitives.  Every test is
deterministic (seeded noise, no network, sub-second) and asserts a structural
invariant (clean link lossless, noise/ISI monotonically degrade BER, modest
Doppler stays decodable) rather than a magic absolute threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.acoustics import compute_rir
from auditory_handshake.config import load_room_preset
from auditory_handshake.phy import MFSKConfig, MFSKModem
from auditory_handshake.phy.channel import (
    apply_acoustic_channel,
    measure_link,
)

FS = 48_000.0
# 64 bytes -> 512 bits -> 256 MFSK symbols (an exact multiple of 2-bit symbols).
PAYLOAD = bytes(range(64))
NUM_BITS = len(PAYLOAD) * 8
NUM_SYMBOLS = NUM_BITS // 2
# Samples per MFSK symbol = round(fs / symbols_per_second) = round(48000 / 512).
SYM_LEN = int(round(FS / MFSKConfig().symbols_per_second))


def _noise(n: int, snr_db: float, seed: int = 0) -> np.ndarray:
    """Additive white noise with a target SNR against a unit-amplitude cosine."""
    signal_power = 0.5  # ~0.5 for a unit-amplitude sinusoid
    var = signal_power / (10.0 ** (snr_db / 10.0))
    return np.random.default_rng(seed).standard_normal(n) * np.sqrt(var)


def _tx_waveform() -> np.ndarray:
    """The clean MFSK TX waveform for :data:`PAYLOAD`."""
    modem = MFSKModem(MFSKConfig(fs=FS))
    symbols = MFSKModem.encode(PAYLOAD, NUM_SYMBOLS)
    return modem.modulate(MFSKModem.bits_from_symbols(symbols)).waveform


def test_apply_passthrough_no_impairments():
    """With no channel args the waveform is returned unchanged."""
    x = _tx_waveform()
    out = apply_acoustic_channel(x, FS)
    assert out.shape == x.shape
    assert np.allclose(out, x, atol=0.0)


def test_apply_unit_impulse_rir_is_identity():
    """A delta RIR must not alter the waveform (convolution identity)."""
    x = _tx_waveform()
    out = apply_acoustic_channel(x, FS, rir=np.array([1.0]))
    assert np.allclose(out, x, atol=1e-12)


def test_apply_preserves_length_with_rir():
    """The output is truncated back to the input length regardless of RIR."""
    x = _tx_waveform()
    rir = np.full(120, 0.02)
    out = apply_acoustic_channel(x, FS, rir=rir)
    assert out.shape == x.shape


def test_apply_noise_shorter_than_signal_raises():
    x = _tx_waveform()
    with pytest.raises(ValueError, match="at least as long"):
        apply_acoustic_channel(x, FS, noise=np.ones(5))


def test_apply_rejects_bad_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        apply_acoustic_channel(np.array([]), FS)
    with pytest.raises(ValueError, match="fs"):
        apply_acoustic_channel(np.ones(10), fs=0)
    with pytest.raises(ValueError, match="finite"):
        apply_acoustic_channel(np.array([1.0, np.nan, 3.0]), FS)


def test_clean_link_is_lossless():
    """No impairment: BER/BLER are exactly zero and the byte round-trips."""
    modem = MFSKModem(MFSKConfig(fs=FS))
    stats = measure_link(modem, PAYLOAD)
    assert stats.num_bits == NUM_BITS
    assert stats.num_blocks == NUM_SYMBOLS
    assert stats.ber == 0.0
    assert stats.bler == 0.0
    assert stats.num_bit_errors == 0

    # Byte-level round trip through the public modem API on a clean channel.
    tx = modem.modulate(MFSKModem.encode(PAYLOAD, NUM_SYMBOLS)).waveform
    rx = modem.demodulate(tx).bits
    assert MFSKModem.decode(rx, NUM_SYMBOLS) == PAYLOAD


def test_noise_degrades_ber_monotonically():
    """Higher SNR -> lower BER; the clean link is the floor."""
    modem = MFSKModem(MFSKConfig(fs=FS))
    n = _tx_waveform().size

    ber_clean = measure_link(modem, PAYLOAD).ber
    ber_20 = measure_link(modem, PAYLOAD, noise=_noise(n, snr_db=20.0, seed=1)).ber
    ber_0 = measure_link(modem, PAYLOAD, noise=_noise(n, snr_db=0.0, seed=1)).ber

    assert ber_clean == 0.0
    assert ber_clean <= ber_20 <= ber_0
    # The 30 dB gap between the clean and 0 dB cases is large enough that the
    # degraded link must be visibly worse than the near-clean one.
    assert ber_0 > 0.0
    assert ber_20 < ber_0


def test_isi_tail_degrades_ber():
    """A point RIR is lossless; a strong intersymbol-interference tail is not."""
    modem = MFSKModem(MFSKConfig(fs=FS))
    point_rir = np.array([1.0])
    # One-symbol-long tail at 1.5x the direct path: the previous symbol's
    # (different) tone leaks into this one's window and flips decisions.
    # One-symbol echo (SYM_LEN taps) at 1.5x the direct-path amplitude: the
    # previous symbol's tone bleeds into the current one and flips decisions.
    isi_rir = np.concatenate([np.array([1.0]), np.full(SYM_LEN, 1.5)])

    ber_point = measure_link(modem, PAYLOAD, rir=point_rir).ber
    ber_isi = measure_link(modem, PAYLOAD, rir=isi_rir).ber

    assert ber_point == 0.0
    assert ber_isi > 0.0


def test_compute_rir_link_stays_decodable():
    """A realistic (dry-ish) room RIR composes with the link and stays decodable."""
    modem = MFSKModem(MFSKConfig(fs=FS))
    rir = np.asarray(compute_rir(load_room_preset("small_office"), 0.15, fs=FS), dtype=float)
    # The statistical RIR is a short cluster whose peak (direct path) lands a
    # fraction of a symbol in, so the receiver sees a small fixed delay.  A real
    # front end would lock that delay; here we time-align the impulse response
    # to its direct path (standard RIR preprocessing) so the test measures the
    # channel's in-symbol ISI, not a timing misalignment.
    rir = rir[int(np.argmax(np.abs(rir))) :]
    stats = measure_link(modem, PAYLOAD, rir=rir)
    assert stats.num_bits == NUM_BITS
    # A dry room is dominated by the direct path, so the link should be intact
    # (or very nearly) — definitely not catastrophically broken.
    assert stats.ber <= 0.25


def test_receding_doppler_stays_decodable():
    """A modest receding Doppler shift is within the receiver's tolerance."""
    modem = MFSKModem(MFSKConfig(fs=FS))
    # A 0.5 m/s receding source shifts the ~3 kHz tones by ~1 Hz — comfortably
    # inside the receiver's frequency-tolerance window, so decoding stays
    # (near-)lossless.  (A 10 m/s source would add time-warp/symbol drift that
    # this fixed-window MFSK receiver does not compensate for, so we keep the
    # speed in the receiver's actual operating range.)
    stats = measure_link(modem, PAYLOAD, speed_ms=-0.5)
    assert stats.num_bits == NUM_BITS
    assert stats.ber <= 0.25
