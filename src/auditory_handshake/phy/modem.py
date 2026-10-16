"""MFSK modem: the physical-layer primitive for the handshake protocol (S13.1).

Multi-frequency shift keying (MFSK) is the natural physical layer for this
project's handshake protocol: each symbol is one pure tone drawn from a
small, pre-agreed constellation, so a receiver only needs an FFT and an
argmax to decode — no pilot tracking, no channel estimation, and the
constant-amplitude, narrow-band symbols are robust to the
frequency-selective fades and Doppler shifts that dominate in
drone scenarios.

Design choices (S13.1 scope: modem only — no channel model, no protocol):

* **Constellation** — 4 tones (4-FSK, 2 bits per symbol) at
  ``fs_hz + k * symbols_per_second`` for ``k = 0..3``.  The spacing equals
  the symbol rate, so the tones are orthogonal over one symbol interval.
* **Offset-blind receiver** — instead of a coherent matched filter (which
  assumes an exact frequency reference), the receiver sums FFT energy in a
  small window around each expected tone position.  A small Doppler shift
  moves the tone within the tolerance window and the decision is
  unchanged (see ``freq_tolerance_bins``).
* **No channel model** — the modem works on clean (or externally noised)
  mono waveforms; adding room RIRs, Doppler, and noise models is S13.2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DEFAULT_FS",
    "DEFAULT_FS_HZ",
    "DEFAULT_SYMBOLS_PER_SECOND",
    "MFSKConfig",
    "MFSKModem",
    "MFSKRxResult",
    "MFSKTxOutput",
]

#: Default sampling rate (Hz) — matches the room model / device presets.
DEFAULT_FS: float = 48_000.0
#: Default base tone frequency (Hz) — inside the audibility window.
DEFAULT_FS_HZ: float = 2_300.0
#: Default symbol rate (symbols/s); also the tone spacing in Hz.
DEFAULT_SYMBOLS_PER_SECOND: int = 512

#: Bits carried by one MFSK symbol (4 tones = 2 bits).
BITS_PER_SYMBOL = 2
#: Number of tones in the constellation.
NUM_TONES = 4


@dataclass(frozen=True)
class MFSKConfig:
    """Modulation parameters for the 4-FSK modem.

    Attributes:
        fs: Sampling rate in Hz.
        fs_hz: Base tone frequency in Hz (the tone for symbol index 0).
        symbols_per_second: Symbol rate in symbols/s.  Also sets the tone
            spacing: tone ``k`` sits at ``fs_hz + k * symbols_per_second``.
        freq_tolerance_bins: Half-width (in FFT bins) of the receiver's
            search window around each expected tone position.  Widen it to
            tolerate Doppler; keep it below half the tone spacing in bins,
            or the windows start to overlap and decisions degrade.
    """

    fs: float = DEFAULT_FS
    fs_hz: float = DEFAULT_FS_HZ
    symbols_per_second: int = DEFAULT_SYMBOLS_PER_SECOND
    freq_tolerance_bins: int = 2
    #: Optional channel impulse response (RIR) for receive-side ISI
    #: equalization.  When set, :meth:`MFSKModem.demodulate` first applies a
    #: frequency-domain MMSE (Wiener) deconvolution by this RIR, removing the
    #: inter-symbol interference the RIR's reverberant tail induces across the
    #: narrow MFSK tones.  ``None`` (the default) disables equalization and
    #: preserves the clean / additive-noise receiver behaviour.
    eq_rir: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.fs < 1.0:
            raise ValueError(f"fs must be >= 1 Hz, got {self.fs}")
        if self.fs_hz < 1.0 or self.fs_hz >= self.fs / 2.0:
            raise ValueError(f"fs_hz must lie in [1, fs/2) Hz, got {self.fs_hz} (fs={self.fs})")
        if self.symbols_per_second < 1:
            raise ValueError(f"symbols_per_second must be >= 1, got {self.symbols_per_second}")
        if self.freq_tolerance_bins < 0:
            raise ValueError(f"freq_tolerance_bins must be >= 0, got {self.freq_tolerance_bins}")
        top_tone = self.fs_hz + (NUM_TONES - 1) * self.symbols_per_second
        if top_tone >= self.fs / 2.0:
            raise ValueError(
                f"highest tone {top_tone} Hz exceeds Nyquist {self.fs / 2.0:.1f} Hz; "
                "lower fs_hz or symbols_per_second"
            )


@dataclass(frozen=True)
class MFSKTxOutput:
    """Result of :meth:`MFSKModem.modulate`."""

    #: Mono waveform (float64), ``len(bits) / 2`` symbols long.
    waveform: np.ndarray
    config: MFSKConfig


@dataclass(frozen=True)
class MFSKRxResult:
    """Result of :meth:`MFSKModem.demodulate`."""

    #: The (validated) input waveform.
    waveform: np.ndarray
    #: Decoded tone indices (int64), one per received symbol, in ``0..3``.
    #: This is the same symbol-index stream that :meth:`MFSKModem.encode`
    #: produces and :meth:`MFSKModem.decode` consumes (a "bit" here means a
    #: symbol as carried on the wire, not a 0/1 data bit).
    bits: np.ndarray
    config: MFSKConfig


def _as_valid_symbols(symbols: np.ndarray | Sequence[int]) -> np.ndarray:
    """Validate a 1-D array of tone indices in ``0..NUM_TONES - 1``."""
    s = np.asarray(symbols, dtype=np.int64)
    if s.ndim != 1:
        raise ValueError(f"symbols must be 1-D, got shape {s.shape}")
    if s.size and (int(s.min()) < 0 or int(s.max()) > NUM_TONES - 1):
        raise ValueError(
            f"symbol indices must be in 0..{NUM_TONES - 1}, got [{int(s.min())}, {int(s.max())}]"
        )
    return s


class MFSKModem:
    """4-FSK modem with an offset-blind (Doppler-tolerant) receiver.

    Two consecutive data bits (MSB first) select one of the four tones
    (index ``0..3``) for a symbol; the receiver finds the tone with the
    most FFT energy near its expected position.
    """

    def __init__(self, config: MFSKConfig | None = None) -> None:
        self._config = config if config is not None else MFSKConfig()

    @property
    def config(self) -> MFSKConfig:
        """The :class:`MFSKConfig` this modem was built with."""
        return self._config

    # -- bit <-> symbol helpers (stateless) --------------------------------

    @staticmethod
    def symbols_from_bits(bits: np.ndarray | Sequence[int]) -> np.ndarray:
        """Pack data bits into tone indices (2 bits per symbol, MSB first)."""
        b = np.asarray(bits, dtype=np.int64)
        if b.ndim != 1 or b.size % BITS_PER_SYMBOL != 0:
            raise ValueError(f"bits must be 1-D with even length, got shape {b.shape}")
        return (b[0::2] * 2 + b[1::2]).astype(np.int64)

    @staticmethod
    def bits_from_symbols(symbols: np.ndarray | Sequence[int]) -> np.ndarray:
        """Expand tone indices back into 2 data bits per symbol (MSB first)."""
        s = _as_valid_symbols(symbols)
        out = np.empty(2 * s.size, dtype=np.int64)
        out[0::2] = (s >> 1) & 1
        out[1::2] = s & 1
        return out

    @staticmethod
    def encode(payload: bytes, num_symbols: int) -> list[int]:
        """Encode ``payload`` as exactly ``num_symbols`` 4-FSK symbols (MSB first).

        The payload must fit within ``num_symbols * 2`` bits.  If it is shorter,
        the trailing bits are zero-padded (whole trailing symbols become ``0``).
        """
        n_bits = num_symbols * BITS_PER_SYMBOL
        if num_symbols < 1:
            raise ValueError(f"num_symbols must be >= 1, got {num_symbols}")
        if len(payload) * 8 > n_bits:
            raise ValueError(
                f"payload ({len(payload)} bytes) does not fit in {num_symbols} symbols"
            )
        bit_list: list[int] = []
        for byte in payload:
            bit_list.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
        bit_list.extend([0] * (n_bits - len(bit_list)))
        bits = np.asarray(bit_list, dtype=np.int64)
        return [int(sym) for sym in MFSKModem.symbols_from_bits(bits)]

    @staticmethod
    def decode(symbols: np.ndarray | Sequence[int], num_symbols: int) -> bytes:
        """Decode symbol indices (from :meth:`demodulate`) back into a byte payload.

        ``symbols`` must contain exactly ``num_symbols`` tone indices in ``0..3``.
        """
        s = _as_valid_symbols(symbols)
        if s.size != num_symbols:
            raise ValueError(f"expected {num_symbols} symbols, got {s.size}")
        bits = MFSKModem.bits_from_symbols(s)
        return np.packbits((bits & 1).astype(np.uint8)).tobytes()

    # -- modulation ---------------------------------------------------------

    def modulate(self, bits: np.ndarray | Sequence[int]) -> MFSKTxOutput:
        """Modulate data into a 4-FSK waveform.

        Accepts either:

        * **bits** — a 1-D array of 0/1 values with even length; each pair of
          bits (MSB first) selects one tone index. Detected when every value
          is ``0`` or ``1`` (e.g. a raw bit stream).
        * **symbols** — a 1-D array of tone indices ``0..3``; detected when
          any value is greater than ``1`` (e.g. the output of :meth:`encode`).

        Args:
            bits: 1-D array of raw bits (0/1) or symbol indices (0..3).

        Returns:
            :class:`MFSKTxOutput` with a mono waveform at ``config.fs``.
            Each symbol occupies ``round(fs / symbols_per_second)`` samples,
            so the waveform length is a multiple of that symbol duration.
        """
        data = np.asarray(bits, dtype=np.int64)
        if data.ndim != 1 or data.size == 0:
            raise ValueError("input must be a non-empty 1-D array of bits or symbols")
        if np.any(data < 0):
            raise ValueError("values must be non-negative")
        if np.any(data > 1):
            # Tone indices (0..3) — validate the range.
            symbols = _as_valid_symbols(data)
        else:
            # Raw bits (0/1) — must be an even number.
            if data.size % BITS_PER_SYMBOL != 0:
                raise ValueError(f"bit input must have an even length, got {data.size}")
            symbols = MFSKModem.symbols_from_bits(data)
        if symbols.size == 0:
            raise ValueError("at least one symbol is required")
        c = self._config
        w = max(1, int(round(c.fs / c.symbols_per_second)))
        t = np.arange(w, dtype=np.float64) / c.fs
        freqs = c.fs_hz + symbols.astype(np.float64)[:, None] * c.symbols_per_second
        tone = 2.0 * np.pi * freqs * t[None, :]
        waveform = np.cos(tone).reshape(symbols.size * w)
        return MFSKTxOutput(waveform=waveform, config=c)

    # -- demodulation --------------------------------------------------------

    def demodulate(self, waveform: np.ndarray | Sequence[float]) -> MFSKRxResult:
        """Demodulate a mono waveform back into symbol indices.

        The signal is split into windows of ``round(fs / symbols_per_second)``
        samples; in each window the FFT energy near each expected tone
        position (within ``freq_tolerance_bins`` bins) is summed and the
        largest wins.  This makes the decision robust to small frequency
        shifts (Doppler) without a frequency reference.

        A trailing partial window (fewer samples than one symbol) is ignored.

        The resulting :attr:`MFSKRxResult.bits` holds one tone index per symbol
        (the same symbol-index stream produced by :meth:`encode`), so it can be
        fed straight back into :meth:`decode`.
        """
        x = np.asarray(waveform, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            raise ValueError(f"waveform must be a non-empty 1-D array, got shape {x.shape}")
        c = self._config
        x = self._equalize(x)
        w = max(1, int(round(c.fs / c.symbols_per_second)))
        n_sym = x.size // w
        if n_sym == 0:
            raise ValueError(
                f"waveform too short for one symbol: fs={c.fs}, "
                f"symbols_per_second={c.symbols_per_second}, length={x.size}"
            )
        symbols = np.zeros(n_sym, dtype=np.int64)
        for i in range(n_sym):
            spec = np.abs(np.fft.rfft(x[i * w : (i + 1) * w])) ** 2
            best_k = 0
            best_metric = -1.0
            for k in range(NUM_TONES):
                metric = self._tone_window_energy(spec, k, w)
                if metric > best_metric:
                    best_metric, best_k = metric, k
            symbols[i] = best_k
        return MFSKRxResult(waveform=x, bits=symbols, config=c)

    # -- convenience ----------------------------------------------------------

    def ber(
        self, bits: np.ndarray | Sequence[int], waveform: np.ndarray | Sequence[float]
    ) -> float:
        """Bit error rate of ``waveform`` against the expected ``bits``.

        ``bits`` may be a raw bit stream (2 per symbol, 0/1) or a symbol-index
        stream (1 per symbol, 0..3, e.g. from :meth:`encode`). The decoded
        symbols are expanded to bits when needed before comparing.
        """
        expected = np.asarray(bits, dtype=np.int64).ravel()
        rx_symbols = self.demodulate(waveform).bits
        if expected.size == rx_symbols.size:
            decoded = rx_symbols
        elif expected.size == BITS_PER_SYMBOL * rx_symbols.size:
            decoded = MFSKModem.bits_from_symbols(rx_symbols)
        else:
            raise ValueError(
                f"length mismatch: expected {expected.size}, " f"decoded {rx_symbols.size} symbols"
            )
        return float(np.mean(expected != decoded))

    # -- internals --------------------------------------------------------------

    def _tone_window_energy(self, spec: np.ndarray, k: int, w: int) -> float:
        """Distance-weighted spectral energy in the tolerance window of tone ``k``.

        Bins closer to the tone's centre contribute more (a triangular weight
        that is ``1`` at the centre and ``0`` at the window edge).  This keeps
        the decision robust to small frequency shifts while still discriminating
        neighbouring tones, whose tolerance windows otherwise overlap and tie
        when a tone falls exactly on an FFT bin.
        """
        c = self._config
        bin_width = c.fs / w
        center = (c.fs_hz + k * c.symbols_per_second) / bin_width
        if center >= len(spec):
            return 0.0
        half = max(1, c.freq_tolerance_bins)
        lo = max(1, int(round(center)) - half)
        hi = min(len(spec) - 1, int(round(center)) + half)
        if hi < 1 or lo >= len(spec):
            return 0.0
        bins = np.arange(lo, hi + 1)
        weights = np.clip(1.0 - np.abs(bins - center) / half, 0.0, 1.0)
        return float(np.sum(spec[lo : hi + 1] * weights))

    def _equalize(self, x: np.ndarray) -> np.ndarray:
        """Apply receive-side channel equalization (ISI removal).

        When ``config.eq_rir`` is set, the received signal ``y = x * h + n``
        is deconvolved in the frequency domain with a linear MMSE (Wiener)
        filter::

            X_hat(f) = Y(f) * conj(H(f)) / (|H(f)|^2 + noise_floor)

        This is a frequency-domain linear inverse of the channel.  It suppresses
        the reverberant tail that otherwise leaks energy from adjacent symbols
        into the narrow MFSK tones (inter-symbol interference), while the
        ``noise_floor`` term keeps the filter well-conditioned at spectral
        notches — unlike a plain zero-forcing filter, which would amplify the
        additive noise there.  A single-tap RIR (pure gain) is simply undone;
        with ``eq_rir`` unset the waveform is returned unchanged so the
        existing clean / additive-noise behaviour is preserved.
        """
        h = self._config.eq_rir
        if h is None:
            return x
        h = np.asarray(h, dtype=np.float64).ravel()
        if h.size == 0:
            return x
        if h.size == 1:
            # Single tap = pure gain; undo it directly (it carries no ISI).
            return x / h[0] if h[0] != 0.0 else x
        needed = x.size + h.size - 1
        n_fft = 1 << (needed - 1).bit_length()  # next power of two >= needed
        Y = np.fft.rfft(x, n_fft)
        H = np.fft.rfft(h, n_fft)
        power = np.abs(H) ** 2
        # A small noise floor (a fraction of the peak in-band channel power)
        # keeps the deconvolution well-conditioned: near zero-forcing where the
        # channel is strong, gentle suppression where it is not.
        noise_floor = max(1e-6 * float(np.max(power)), np.finfo(np.float64).eps)
        X = Y * np.conj(H) / (power + noise_floor)
        return np.fft.irfft(X, n_fft)[: x.size]
