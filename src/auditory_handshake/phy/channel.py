"""Acoustic channel effects applied to MFSK waveforms (PLAN S13.2).

The S13.1 modem works on a clean mono waveform.  This module closes the gap to
the real link by pushing a modulated waveform through the acoustic
impairments the PLAN §4 architecture lists — Doppler from source motion, room
reverberation (RIR), and additive microphone noise — and by providing a
BER/BLER measurement over the full
``encode → modulate → channel → demodulate → decode`` path.

It *composes* the existing :mod:`auditory_handshake.acoustics` primitives
(Doppler via :func:`~auditory_handshake.acoustics.moving_source`, plus a
caller-supplied RIR and noise waveform) rather than regenerating them, so this
stays a thin, pure, config-free bridge between the two halves of the system.
The M3 waveform A/B study (``sims/03``) will reuse :func:`measure_link` to
score MFSK against CSS and OFDM-QAM on the same channel.

All functions are pure and stateless; nothing here mutates its inputs or holds
globals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from auditory_handshake.acoustics import moving_source
from auditory_handshake.phy.modem import BITS_PER_SYMBOL, MFSKModem

__all__ = [
    "LinkStats",
    "apply_acoustic_channel",
    "measure_link",
]


@dataclass(frozen=True)
class LinkStats:
    """BER/BLER of one MFSK frame carried across the acoustic channel.

    A *block* is one MFSK symbol (``BITS_PER_SYMBOL`` data bits), so
    :attr:`bler` is the fraction of symbols that contain at least one wrong
    bit and :attr:`ber` the fraction of individual data bits that are wrong.
    ``bler >= ber`` always (a block is wrong whenever any of its bits is).
    """

    #: Total data bits transmitted (``num_symbols * BITS_PER_SYMBOL``).
    num_bits: int
    #: Data bits received incorrectly.
    num_bit_errors: int
    #: Number of symbols (blocks) transmitted.
    num_blocks: int
    #: Symbols (blocks) containing at least one wrong bit.
    num_block_errors: int

    @property
    def ber(self) -> float:
        """Bit error rate in ``0.0 .. 1.0`` (``0.0`` for an empty frame)."""
        return 0.0 if self.num_bits == 0 else self.num_bit_errors / self.num_bits

    @property
    def bler(self) -> float:
        """Block (symbol) error rate in ``0.0 .. 1.0`` (``0.0`` if no blocks)."""
        return 0.0 if self.num_blocks == 0 else self.num_block_errors / self.num_blocks


def apply_acoustic_channel(
    waveform: np.ndarray | Sequence[float],
    fs: float,
    *,
    rir: np.ndarray | Sequence[float] | None = None,
    noise: np.ndarray | Sequence[float] | None = None,
    speed_ms: float | None = None,
    block_duration_seconds: float | None = None,
) -> np.ndarray:
    """Pass ``waveform`` through the acoustic chain: Doppler → RIR → noise.

    Each stage is optional; omit an argument to skip it.  The order matches
    the PLAN §4 acoustic path (loudspeaker → room → microphone): source motion
    first, then reverberation, then additive noise at the receiver.

    Args:
        waveform: 1-D mono signal at sampling rate ``fs`` (e.g. the output of
            :meth:`MFSKModem.modulate`).
        fs: Sampling rate in Hz (``>= 1``).
        rir: Optional 1-D room impulse response.  The post-Doppler waveform is
            linearly convolved with it and truncated back to the original
            length, so the output aligns sample-for-sample with the input, with
            the RIR tail past the window discarded.
        noise: Optional 1-D additive noise; must be at least as long as
            ``waveform``.  Added to the post-RIR signal.
        speed_ms: Optional source closing speed in m/s (``> 0`` approaching,
            ``< 0`` receding).  ``None`` = static.  Approaching motion needs a
            long-enough signal for the source-walk (see
            :func:`auditory_handshake.acoustics.moving_source`).
        block_duration_seconds: Doppler resampling block; defaults to the full
            signal duration (a single, uniform-Doppler block).

    Returns:
        1-D ``float64`` received waveform of the same length as ``waveform``.

    Raises:
        ValueError: If ``fs`` is not positive, ``waveform``/``rir`` is not a
            non-empty 1-D array of finite values, or ``noise`` is too short.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"waveform must be a non-empty 1-D array, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("waveform must contain only finite values")
    if fs < 1.0:
        raise ValueError(f"fs must be >= 1 Hz, got {fs}")

    out = x
    if speed_ms is not None:
        duration = x.size / fs
        block = block_duration_seconds if block_duration_seconds is not None else duration
        out = moving_source(
            out, fs, speed_ms, duration_seconds=duration, block_duration_seconds=block
        )
    if rir is not None:
        r = np.asarray(rir, dtype=np.float64)
        if r.ndim != 1 or r.size == 0:
            raise ValueError(f"rir must be a non-empty 1-D array, got shape {r.shape}")
        if not np.all(np.isfinite(r)):
            raise ValueError("rir must contain only finite values")
        out = np.convolve(out, r, mode="full")[: x.size]
    if noise is not None:
        nz = np.asarray(noise, dtype=np.float64)
        if nz.size < out.size:
            raise ValueError(
                f"noise ({nz.size} samples) must be at least as long as the "
                f"waveform ({out.size} samples)"
            )
        if not np.all(np.isfinite(nz[: out.size])):
            raise ValueError("noise must contain only finite values")
        out = out + nz[: out.size]
    return out.astype(np.float64)


def measure_link(
    modem: MFSKModem,
    payload: bytes,
    *,
    rir: np.ndarray | Sequence[float] | None = None,
    noise: np.ndarray | Sequence[float] | None = None,
    speed_ms: float | None = None,
    block_duration_seconds: float | None = None,
) -> LinkStats:
    """Run the full MFSK link over the acoustic channel and report BER/BLER.

    The flow is ``encode → modulate → apply_acoustic_channel → demodulate``;
    the decoded symbols are compared against the transmitted symbols at both
    the bit and the block (symbol) level.

    Args:
        modem: A :class:`MFSKModem` (its ``config`` supplies the symbol rate,
            tones, and sampling rate for modulation/demodulation).
        payload: The byte string to transmit.  It is encoded into
            ``ceil(len(payload) * 8 / 2)`` symbols, with any trailing bits
            zero-padded to a whole symbol.
        rir, noise, speed_ms, block_duration_seconds: Passed straight through
            to :func:`apply_acoustic_channel` (see its docstring).

    Returns:
        A :class:`LinkStats` with the bit/block error counts and the
        :attr:`~LinkStats.ber` / :attr:`~LinkStats.bler` rates.

    Raises:
        ValueError: If ``payload`` is empty or any channel argument is invalid.
    """
    data = bytes(payload)
    if not data:
        raise ValueError("payload must be non-empty")
    fs = modem.config.fs

    num_symbols = (len(data) * 8 + BITS_PER_SYMBOL - 1) // BITS_PER_SYMBOL
    tx_symbols = np.asarray(MFSKModem.encode(data, num_symbols), dtype=np.int64)
    # Modulate from the *bit* stream so the modem's bit/symbol input
    # disambiguation never misfires on an all-0/1 symbol pattern.
    tx_waveform = modem.modulate(MFSKModem.bits_from_symbols(tx_symbols)).waveform

    rx_waveform = apply_acoustic_channel(
        tx_waveform,
        fs,
        rir=rir,
        noise=noise,
        speed_ms=speed_ms,
        block_duration_seconds=block_duration_seconds,
    )
    rx_symbols = modem.demodulate(rx_waveform).bits

    tx_bits = MFSKModem.bits_from_symbols(tx_symbols).reshape(num_symbols, BITS_PER_SYMBOL)
    rx_bits = MFSKModem.bits_from_symbols(rx_symbols).reshape(num_symbols, BITS_PER_SYMBOL)
    wrong = tx_bits != rx_bits

    return LinkStats(
        num_bits=int(wrong.size),
        num_bit_errors=int(wrong.sum()),
        num_blocks=int(num_symbols),
        num_block_errors=int(wrong.any(axis=1).sum()),
    )
