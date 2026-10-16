"""Tests for the CSS (chirp spread spectrum) modem.

Mirrors :mod:`tests.test_modem` for the MFSK modem.  Uses a low-rate test
configuration (fs = 8 kHz, 100 sps) so that symbol durations and FFT bins are
exact integers, and the chirp constellations are simple to reason about.
"""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.phy import CSSConfig, CSSModem

# 8 kHz sampling, 100 Hz base, 4 chirps, 200 Hz sweep each, 100 sps.
# chirp_spacing = 200/4 = 50 Hz; chirp m sweeps [100 + m*50, 300 + m*50] Hz.
# nsym = fs / (2*sps) = 40; fft_size = 80.
CFG = CSSConfig(
    fs=8_000.0,
    fs_hz=100.0,
    symbols_per_second=100,
    n_chirps=4,
    total_bandwidth=200.0,
    freq_tolerance_bins=1,
)


def _random_symbols(n_symbols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, CFG.n_chirps, size=n_symbols)


def test_config_defaults_valid() -> None:
    cfg = CSSConfig()
    assert cfg.bits_per_symbol == 2
    assert cfg.n_chirps == 4
    # Nyquist guard must hold for the shipped defaults.
    top_edge = cfg.fs_hz + (cfg.n_chirps - 1) * cfg.chirp_spacing + cfg.total_bandwidth
    assert top_edge < cfg.fs / 2.0


def test_each_chirp_demodulates_to_itself() -> None:
    """Each of the n_chirps symbols round-trips to its own peak bin (no collisions)."""
    modem = CSSModem(CFG)
    for sym in range(CFG.n_chirps):
        rx = modem.demodulate(modem.modulate([sym]).waveform)
        assert rx.bits.tolist() == [sym]


def test_clean_roundtrip_recovers_payload() -> None:
    modem = CSSModem(CFG)
    payload = b"handshake"
    num_symbols = len(payload) * 8 // 2  # 36 symbols
    symbols = modem.encode(payload, num_symbols)
    tx = modem.modulate(symbols)
    rx = modem.demodulate(tx.waveform)
    assert rx.bits.tolist() == symbols
    assert modem.decode(rx.bits, num_symbols) == payload


def test_ber_zero_on_clean_waveform() -> None:
    modem = CSSModem(CFG)
    symbols = _random_symbols(64)
    tx = modem.modulate(symbols)
    rx = modem.demodulate(tx.waveform)
    assert modem.ber(symbols, rx.bits) == 0.0


def test_ber_low_at_high_snr() -> None:
    modem = CSSModem(CFG)
    symbols = _random_symbols(128)
    tx = modem.modulate(symbols).waveform
    snr = 12.0
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=np.sqrt(tx.var() / (10.0 ** (snr / 10.0))))
    rx = modem.demodulate(tx + noise)
    assert modem.ber(symbols, rx.bits) < 0.05


def test_symbol_bit_mapping() -> None:
    modem = CSSModem(CFG)
    assert modem.symbols_from_bits([1, 0, 0, 1]).tolist() == [2, 1]
    assert modem.bits_from_symbols([2, 1]).tolist() == [1, 0, 0, 1]


def test_encode_decode_roundtrip_and_padding() -> None:
    modem = CSSModem(CFG)
    symbols = modem.encode(b"ab", 8)  # 8 symbols = 16 bits = 2 bytes
    assert len(symbols) == 8
    assert modem.decode(symbols, 8) == b"ab"
    # Padding: 1 byte needs 4 symbols; request 6 -> two zero symbols at the end.
    symbols2 = modem.encode(b"\x06", 6)
    assert symbols2[4:] == [0, 0]
    assert modem.decode(symbols2[:4], 4) == b"\x06"


def test_validation() -> None:
    modem = CSSModem(CFG)
    with pytest.raises(ValueError):
        modem.modulate([4])  # chirp index out of range for n_chirps=4
    with pytest.raises(ValueError):
        modem.modulate([-1])  # negative chirp index
    with pytest.raises(ValueError):
        modem.symbols_from_bits([0])  # odd bit length
    with pytest.raises(ValueError):
        CSSConfig(n_chirps=3)  # not a power of two
    with pytest.raises(ValueError):
        CSSConfig(fs=8_000.0, fs_hz=4_000.0)  # base tone above Nyquist
    with pytest.raises(ValueError):
        CSSConfig(fs=8_000.0, fs_hz=3_000.0, total_bandwidth=1_000.0)  # top edge past Nyquist
