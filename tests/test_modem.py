"""Tests for the MFSK modem (S13.1, ``src/auditory_handshake/phy/modem.py``).

Uses a low-rate test configuration (fs = 8 kHz, 100 sps) so that symbol
durations and FFT bins are exact integers and tone-placement assertions can
be checked exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.phy import MFSKConfig, MFSKModem

# 8 kHz sampling, 400 Hz base tone, 100 sps -> 10 Hz/bin, tone spacing 100 Hz = 10 bins.
CFG = MFSKConfig(fs=8_000.0, fs_hz=400.0, symbols_per_second=100)


def _random_bits(n_symbols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_symbols * 2)


def test_tone_placement() -> None:
    modem = MFSKModem(CFG)
    # Symbol index 0 -> base tone 400 Hz; index 3 -> 400 + 3*100 = 700 Hz.
    for bits, expected_hz in (([0, 0], 400.0), ([1, 1], 700.0)):
        x = modem.modulate(bits).waveform
        spec = np.abs(np.fft.rfft(x))
        peak_hz = (float(np.argmax(spec[1:])) + 1) * (CFG.fs / len(x))
        assert peak_hz == pytest.approx(expected_hz, rel=1e-9)


def test_clean_roundtrip_recovers_payload() -> None:
    modem = MFSKModem(CFG)
    payload = b"handshake"
    num_symbols = len(payload) * 8 // 2  # 36 symbols
    bits = modem.encode(payload, num_symbols)
    tx = modem.modulate(bits)
    rx = modem.demodulate(tx.waveform)
    assert rx.bits.tolist() == bits
    assert modem.decode(rx.bits, num_symbols) == payload


def test_ber_zero_on_clean_waveform() -> None:
    modem = MFSKModem(CFG)
    bits = _random_bits(64)
    tx = modem.modulate(bits)
    assert modem.ber(bits, tx.waveform) == 0.0


def test_ber_low_at_high_snr() -> None:
    modem = MFSKModem(CFG)
    bits = _random_bits(64)
    tx = modem.modulate(bits)
    snr = 12.0
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=np.sqrt(tx.waveform.var() / (10.0 ** (snr / 10.0))))
    assert modem.ber(bits, tx.waveform + noise) < 0.05


def test_robust_to_frequency_shift() -> None:
    """A small Doppler-like frequency shift stays inside the tolerance window."""
    modem = MFSKModem(CFG)
    bits = _random_bits(32)
    tx = modem.modulate(bits)
    t = np.arange(tx.waveform.size) / CFG.fs
    shift_hz = 5.0  # half an FFT bin (10 Hz); inside the default +/-2-bin window
    shifted = tx.waveform * np.cos(2.0 * np.pi * shift_hz * t)
    assert modem.ber(bits, shifted) == 0.0


def test_symbol_bit_mapping() -> None:
    assert MFSKModem.symbols_from_bits([1, 0, 0, 1]).tolist() == [2, 1]
    assert MFSKModem.bits_from_symbols([2, 1]).tolist() == [1, 0, 0, 1]


def test_encode_decode_roundtrip_and_padding() -> None:
    payload = b"ab"
    bits = MFSKModem.encode(payload, 8)  # 8 symbols = 16 bits = 2 bytes
    assert len(bits) == 8
    assert MFSKModem.decode(bits, 8) == payload
    # Padding: 1 byte needs 4 symbols, but we request 6 -> two zero symbols at the end.
    bits2 = MFSKModem.encode(b"\x06", 6)
    assert bits2[4:] == [0, 0]
    assert MFSKModem.decode(bits2[:4], 4) == b"\x06"


def test_validation() -> None:
    modem = MFSKModem(CFG)
    with pytest.raises(ValueError):
        modem.modulate([1, 1, 1])  # odd number of bits
    with pytest.raises(ValueError):
        MFSKModem.encode(b"ab", 3)  # 3 symbols = 6 bits, not a byte multiple
    with pytest.raises(ValueError):
        MFSKModem.encode(b"abcdef", 4)  # 3 bytes do not fit in 8 bits
    with pytest.raises(ValueError):
        MFSKModem.symbols_from_bits([0])  # odd
    with pytest.raises(ValueError):
        MFSKModem.bits_from_symbols([4])  # index out of range
    with pytest.raises(ValueError):
        MFSKConfig(fs=8_000.0, fs_hz=4_001.0)  # base tone above Nyquist
    with pytest.raises(ValueError):
        MFSKConfig(fs=8_000.0, fs_hz=4_000.0, symbols_per_second=700)  # top tone past Nyquist
