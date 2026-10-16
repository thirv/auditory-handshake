"""CSS (chirp spread spectrum) modem (S13.3, M3).

CSS transmits each symbol as a **linear-frequency-modulated (LFM) chirp** that
sweeps the whole configured bandwidth during one symbol interval, rather than
as a single constant-frequency tone like :mod:`auditory_handshake.phy.modem`
(MFSK/FSK).  Data is carried in the *starting frequency* of the chirp: symbol
``m`` begins at ``fs_hz + m * chirp_spacing`` and sweeps upward by
``total_bandwidth`` Hz over the symbol.  The chirps are grouped in
``bits_per_symbol = log2(n_chirps)`` data bits (MSB first); the default (4
chirps) carries 2 bits/symbol, the same as the 4-FSK modem, which keeps the
CSS-vs-MFSK A/B comparison (S13.6) fair.

Why a different waveform for the same acoustic channel (PLAN.md §2)?

* A CSS chirp spreads each symbol's energy across ``total_bandwidth`` Hz, so a
  narrowband notch or a single deep fade knocks out only part of a symbol and
  the de-chirp receiver recovers the rest (processing gain / frequency
  diversity).
* MFSK concentrates each symbol in one narrow tone: cheap and
  timing-tolerant, but a fade or narrowband interference on that one bin loses
  the whole symbol.
* CSS chirps are wideband, so they are more sensitive to inter-symbol
  interference; MFSK tones are narrowband and more ISI-tolerant.

That contrast is what S13.6's A/B test measures, and it is why CSS is planned
for the higher, less-audible 10–20 kHz band (``configs/defaults.yaml::
audio_bands_hz``) while MFSK targets 8–12 kHz.

Receiver (de-chirp + windowed FFT):  for each symbol window the common chirp
slope is removed (multiply by the conjugate of a zero-offset LFM reference),
collapsing the transmitted chirp into a pure tone at its start frequency.  A
per-candidate-tone, distance-weighted FFT-energy test (the same idiom as the
FSK receiver) then picks the chirp whose start frequency has the most energy.
Because a delay maps to a small frequency offset under de-chirp, the tolerance
window makes CSS *similarly* robust to small delay/Doppler as MFSK — the A/B
difference is the on-air waveform, not the receiver's tolerance.

The public interface mirrors :class:`~auditory_handshake.phy.modem.MFSKModem`
(``config`` / ``encode`` / ``decode`` / ``symbols_from_bits`` /
``bits_from_symbols`` / ``modulate`` / ``demodulate`` / ``ber``) so the S13.6
simulation can drive either modem through the same ``modulate →
apply_acoustic_channel → demodulate → compare`` path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# --- defaults ----------------------------------------------------------------
# The CSS waveform targets the 10–20 kHz band declared in configs/defaults.yaml
# (audio_bands_hz.css), i.e. *above* the MFSK 8–12 kHz band and close to the
# upper limit of human audibility — the PLAN's "less pleasant, more bandwidth"
# trade.  Defaults below keep the constellation inside that band: each chirp
# sweeps ``total_bandwidth`` Hz and is offset by ``chirp_spacing`` Hz, so the
# highest chirp's top edge is
# ``fs_hz + (n_chirps - 1) * chirp_spacing + total_bandwidth`` =
# ``fs_hz + (2 - 1/n_chirps) * total_bandwidth``; with the values here that is
# 11 kHz + 1.75 * 4 kHz = 18 kHz, below both the 20 kHz band edge and Nyquist.
DEFAULT_FS = 48_000.0
DEFAULT_FS_HZ = 11_000.0
DEFAULT_SYMBOLS_PER_SECOND = 512
DEFAULT_N_CHIRPS = 4
DEFAULT_TOTAL_BANDWIDTH = 4_000.0


def _bits_per_symbol(n_chirps: int) -> int:
    """Number of data bits encoded by ``n_chirps`` chirps (log2)."""
    return int(round(math.log2(n_chirps)))


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _as_valid_symbols(symbols: np.ndarray | Sequence[int], n_chirps: int) -> np.ndarray:
    """Validate a 1-D array of chirp indices in ``0..n_chirps - 1``."""
    s = np.asarray(symbols, dtype=np.int64)
    if s.ndim != 1:
        raise ValueError(f"symbols must be 1-D, got shape {s.shape}")
    if s.size and (int(s.min()) < 0 or int(s.max()) > n_chirps - 1):
        raise ValueError(
            f"symbol indices must be in 0..{n_chirps - 1}, got [{int(s.min())}, {int(s.max())}]"
        )
    return s


@dataclass(frozen=True)
class CSSConfig:
    """Modulation parameters for the LFM-chirp (CSS) modem.

    Attributes:
        fs: Sampling rate in Hz.
        fs_hz: Start frequency (Hz) of the *lowest* chirp (symbol index 0).
        symbols_per_second: Symbol rate in symbols/s.  Each symbol occupies
            ``round(fs / symbols_per_second)`` samples and its chirp sweeps
            ``total_bandwidth`` Hz over that interval.
        n_chirps: Number of chirps in the constellation.  Must be a power of
            two >= 2; ``log2(n_chirps)`` data bits are carried per symbol.
        total_bandwidth: Total spectral span (Hz) each chirp sweeps.  Chirp
            ``m`` sweeps ``[fs_hz + m·(total_bandwidth/n_chirps),
            fs_hz + m·(total_bandwidth/n_chirps) + total_bandwidth]``.
        freq_tolerance_bins: Half-width (in FFT bins) of the receiver's search
            window around each candidate start frequency.  Widen to tolerate
            Doppler/delay; keep it below half the chirp spacing in bins so the
            decision windows do not overlap too much.
    """

    fs: float = DEFAULT_FS
    fs_hz: float = DEFAULT_FS_HZ
    symbols_per_second: int = DEFAULT_SYMBOLS_PER_SECOND
    n_chirps: int = DEFAULT_N_CHIRPS

    #: Optional room-impulse response (RIR) used to equalize the received
    #: waveform in :meth:`demodulate` (frequency-domain MMSE deconvolution),
    #: mirroring :attr:`MFSKConfig.eq_rir` so the S13.6 A/B comparison is fair
    #: when both modems are driven through a reverberant acoustic channel.
    #: ``None`` (the default) disables equalization.
    eq_rir: np.ndarray | None = None
    total_bandwidth: float = DEFAULT_TOTAL_BANDWIDTH
    freq_tolerance_bins: int = 2

    def __post_init__(self) -> None:
        if self.fs < 1.0:
            raise ValueError(f"fs must be >= 1 Hz, got {self.fs}")
        if self.fs_hz < 1.0 or self.fs_hz >= self.fs / 2.0:
            raise ValueError(f"fs_hz must lie in [1, fs/2) Hz, got {self.fs_hz} (fs={self.fs})")
        if self.symbols_per_second < 1:
            raise ValueError(f"symbols_per_second must be >= 1, got {self.symbols_per_second}")
        if not _is_power_of_two(self.n_chirps) or self.n_chirps < 2:
            raise ValueError(f"n_chirps must be a power of two >= 2, got {self.n_chirps}")
        if self.total_bandwidth < 1.0:
            raise ValueError(f"total_bandwidth must be >= 1 Hz, got {self.total_bandwidth}")
        if self.freq_tolerance_bins < 0:
            raise ValueError(f"freq_tolerance_bins must be >= 0, got {self.freq_tolerance_bins}")
        top_edge = self.fs_hz + (self.n_chirps - 1) * self.chirp_spacing + self.total_bandwidth
        if top_edge >= self.fs / 2.0:
            raise ValueError(
                f"highest chirp edge {top_edge:.1f} Hz exceeds Nyquist "
                f"{self.fs / 2.0:.1f} Hz; lower fs_hz or total_bandwidth"
            )

    @property
    def bits_per_symbol(self) -> int:
        """Data bits carried per symbol (``log2(n_chirps)``)."""
        return _bits_per_symbol(self.n_chirps)

    @property
    def chirp_spacing(self) -> float:
        """Hz separating the start frequency of adjacent chirps."""
        return self.total_bandwidth / self.n_chirps


@dataclass(frozen=True)
class CSSTxOutput:
    """Result of :meth:`CSSModem.modulate`.

    Attributes:
        waveform: Mono waveform (float64), ``round(fs / symbols_per_second)``
            samples per symbol.
        config: The :class:`CSSConfig` used for the modulation.
    """

    waveform: np.ndarray
    config: CSSConfig


@dataclass(frozen=True)
class CSSRxResult:
    """Result of :meth:`CSSModem.demodulate`.

    Attributes:
        waveform: The (validated) input waveform.
        bits: Decoded chirp indices (int64), one per received symbol, in
            ``0..n_chirps-1``.  Same symbol-index stream :meth:`CSSModem.encode`
            produces and :meth:`CSSModem.decode` consumes.
        config: The :class:`CSSConfig` used for the demodulation.
    """

    waveform: np.ndarray
    bits: np.ndarray
    config: CSSConfig


class CSSModem:
    """LFM-chirp (CSS) modem with a de-chirp + windowed-FFT receiver.

    ``bits_per_symbol`` (default 2, i.e. 4 chirps) consecutive data bits
    (MSB first) select one chirp; the receiver de-chirps each symbol window
    into a pure tone and picks the chirp whose start frequency has the most
    FFT energy (with a distance-weighted tolerance window).
    """

    def __init__(self, config: CSSConfig | None = None) -> None:
        self._config = config if config is not None else CSSConfig()

    @property
    def config(self) -> CSSConfig:
        """The :class:`CSSConfig` this modem was built with."""
        return self._config

    @property
    def bits_per_symbol(self) -> int:
        """Data bits carried per symbol (``log2(n_chirps)``)."""
        return self._config.bits_per_symbol

    @property
    def samples_per_symbol(self) -> int:
        """Samples per symbol (``round(fs / symbols_per_second)``).

        The same value :meth:`modulate` uses to size each chirp window, so
        callers can compute a symbol slice length (``num_symbols *
        samples_per_symbol``) without duplicating that formula.
        """
        return int(round(self._config.fs / self._config.symbols_per_second))

    # -- bit <-> symbol mapping ---------------------------------------------

    def symbols_from_bits(self, bits: np.ndarray | Sequence[int]) -> np.ndarray:
        """Group raw data bits (MSB first) into chirp indices ``0..n_chirps-1``.

        Args:
            bits: 1-D array of 0/1 values with length a multiple of
                ``bits_per_symbol``.
        Returns:
            int64 array of chirp indices, length ``len(bits) // bits_per_symbol``.
        """
        b = np.asarray(bits, dtype=np.int64)
        if b.ndim != 1:
            raise ValueError(f"bits must be 1-D, got shape {b.shape}")
        bps = self._config.bits_per_symbol
        if b.size % bps != 0:
            raise ValueError(f"bits length must be a multiple of {bps}, got {b.size}")
        b = b & 1
        groups = b.reshape(-1, bps)
        weights = (1 << np.arange(bps - 1, -1, -1)).astype(np.int64)
        symbols = groups @ weights
        if int(symbols.max(initial=0)) > self._config.n_chirps - 1:
            raise ValueError("bit groups exceed n_chirps - 1")
        return symbols.astype(np.int64)

    def bits_from_symbols(self, symbols: np.ndarray | Sequence[int]) -> np.ndarray:
        """Expand chirp indices ``0..n_chirps-1`` into raw data bits (MSB first).

        Args:
            symbols: 1-D array of chirp indices in ``0..n_chirps - 1``.
        Returns:
            int64 bit array, length ``len(symbols) * bits_per_symbol``.
        """
        bps = self._config.bits_per_symbol
        s = _as_valid_symbols(symbols, self._config.n_chirps)
        mask = np.arange(bps - 1, -1, -1, dtype=np.int64)
        bits = (s.astype(np.int64)[:, None] >> mask[None, :]) & 1
        return bits.reshape(-1)

    def encode(self, payload: bytes | bytearray | int, num_symbols: int) -> list[int]:
        """Encode ``payload`` into exactly ``num_symbols`` chirp indices.

        Bytes are expanded MSB-first; the bit stream is zero-padded (or the
        payload truncated) to match the symbol count.  An ``int`` payload is
        treated as the value of its binary expansion (most significant bit
        first) and must fit.

        Args:
            payload: ``bytes``/``bytearray`` (MSB-first) or a non-negative ``int``.
            num_symbols: Number of chirp symbols to emit (>= 0).
        Returns:
            A list of ``num_symbols`` chirp indices in ``0..n_chirps-1``.
        """
        if num_symbols < 0:
            raise ValueError(f"num_symbols must be >= 0, got {num_symbols}")
        bps = self._config.bits_per_symbol
        n_bits = num_symbols * bps
        if n_bits == 0:
            if isinstance(payload, int):
                if payload != 0:
                    raise ValueError(f"cannot encode {payload} into 0 symbols")
            elif payload:
                raise ValueError(f"cannot encode {len(payload)} byte(s) into 0 symbols")
            return []
        bit_list: list[int]
        if isinstance(payload, int):
            if payload < 0:
                raise ValueError(f"payload int must be >= 0, got {payload}")
            if payload.bit_length() > n_bits:
                raise ValueError(
                    f"payload 0x{payload:x} needs more than {n_bits} bits "
                    f"({num_symbols} symbols x {bps} bits)"
                )
            bit_list = [(payload >> shift) & 1 for shift in range(n_bits - 1, -1, -1)]
        else:
            bit_list = []
            for byte in payload:
                bit_list.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
            if len(bit_list) > n_bits:
                bit_list = bit_list[:n_bits]
            bit_list.extend([0] * (n_bits - len(bit_list)))
        bits = np.asarray(bit_list, dtype=np.int64)
        return [int(sym) for sym in self.symbols_from_bits(bits)]

    def decode(self, symbols: np.ndarray | Sequence[int], num_symbols: int) -> bytes:
        """Decode chirp indices (from :meth:`demodulate`) back into a byte payload.

        Args:
            symbols: Exactly ``num_symbols`` chirp indices in ``0..n_chirps-1``.
            num_symbols: Number of chirp symbols being decoded.
        Returns:
            The reassembled ``bytes`` (trailing zero bits discarded).
        """
        s = _as_valid_symbols(symbols, self._config.n_chirps)
        if s.size != num_symbols:
            raise ValueError(f"expected {num_symbols} symbols, got {s.size}")
        bits = self.bits_from_symbols(s)
        return np.packbits((bits & 1).astype(np.uint8)).tobytes()

    # -- modulation -----------------------------------------------------------

    def modulate(self, symbols: np.ndarray | Sequence[int]) -> CSSTxOutput:
        """Synthesize a mono waveform from chirp indices.

        Each chirp is a unit-amplitude LFM sweep: chirp ``m`` starts at
        ``fs_hz + m * chirp_spacing`` and sweeps ``total_bandwidth`` Hz over
        ``round(fs / symbols_per_second)`` samples.  Symbols are concatenated
        one symbol-interval per chirp.

        Args:
            symbols: 1-D array of chirp indices in ``0..n_chirps - 1``.

        Returns:
            :class:`CSSTxOutput` with the concatenated mono ``waveform``.

        Raises:
            ValueError: If any index is outside ``0..n_chirps - 1`` or the
                ``symbols`` array is not 1-D.
        """
        cfg = self._config
        s = _as_valid_symbols(symbols, cfg.n_chirps)
        if s.size == 0:
            return CSSTxOutput(waveform=np.zeros(0, dtype=np.float64), config=cfg)

        nsym = int(round(cfg.fs / cfg.symbols_per_second))
        t = np.arange(nsym, dtype=np.float64) / cfg.fs
        # Chirp rate (Hz/s): total_bandwidth swept over one symbol interval.
        K = cfg.total_bandwidth / nsym
        # Zero-offset LFM phase reference (shared by TX and the de-chirp RX).
        ref = np.exp(1j * np.pi * K * t**2)

        spacing = cfg.chirp_spacing
        # Precompute each chirp's waveform (n_chirps is small: 2, 4, 8, 16).
        per_chirp = np.empty((cfg.n_chirps, nsym), dtype=np.complex128)
        for m in range(cfg.n_chirps):
            f0 = cfg.fs_hz + m * spacing
            per_chirp[m] = np.exp(2j * np.pi * f0 * t) * ref

        waveform = np.empty(s.size * nsym, dtype=np.float64)
        for i, m in enumerate(s):
            waveform[i * nsym : (i + 1) * nsym] = per_chirp[int(m)].real
        return CSSTxOutput(waveform=waveform, config=cfg)

    # -- demodulation ---------------------------------------------------------

    def demodulate(self, waveform: np.ndarray | Sequence[float]) -> CSSRxResult:
        """Recover chirp indices from a (possibly noisy) waveform.

        The waveform is chunked into ``round(fs / symbols_per_second)``-sample
        symbol windows.  For each window the common chirp slope is removed
        (multiply by the conjugate of the zero-offset LFM reference), which
        collapses each chirp into a pure tone at its start frequency; a
        distance-weighted FFT-energy test over the chirp grid then picks the
        best chirp.  A window shorter than the symbol interval (a truncated
        symbol at the very end of the signal) is still decoded best-effort.

        Args:
            waveform: 1-D real signal in Hz-domain samples.

        Returns:
            :class:`CSSRxResult` with the decoded chirp indices in ``bits``
            (one per symbol window).

        Raises:
            ValueError: If ``waveform`` is not 1-D.
        """
        cfg = self._config
        wf = np.asarray(waveform, dtype=np.float64)
        if wf.ndim != 1:
            raise ValueError(f"waveform must be 1-D, got shape {wf.shape}")

        nsym = int(round(cfg.fs / cfg.symbols_per_second))
        n_full = wf.size // nsym
        if n_full == 0:
            return CSSRxResult(waveform=wf, bits=np.zeros(0, dtype=np.int64), config=cfg)

        wf = self._equalize(wf)

        t = np.arange(nsym, dtype=np.float64) / cfg.fs
        K = cfg.total_bandwidth / nsym
        ref = np.exp(1j * np.pi * K * t**2)
        spacing = cfg.chirp_spacing
        n_chirps = cfg.n_chirps
        tol = cfg.freq_tolerance_bins
        fft_size = 2 * tol * nsym

        bits = np.empty(n_full, dtype=np.int64)
        for i in range(n_full):
            seg = wf[i * nsym : (i + 1) * nsym]
            # De-chirp: remove the common LFM slope -> pure tone at start freq.
            dechirped = np.asarray(seg, dtype=np.complex128) * np.conj(ref)
            dechirped = np.fft.fftshift(np.fft.fft(dechirped, n=fft_size))
            mag2 = (np.abs(dechirped) ** 2).astype(np.float64)

            best_symbol = 0
            best_score = -np.inf
            for m in range(n_chirps):
                f0 = cfg.fs_hz + m * spacing
                # After fftshift, index j maps to frequency (j - n/2) * fs/n,
                # so a positive tone at f0 lands at index n/2 + round(n*f0/fs).
                center = fft_size // 2 + int(round(fft_size * f0 / cfg.fs))
                lo = max(center - tol, 0)
                hi = min(center + tol, fft_size - 1)
                window = mag2[lo : hi + 1]
                if window.size == 0:
                    continue
                center_pos = center - lo
                if center_pos < 0 or center_pos >= window.size:
                    score = float(np.sum(window))
                else:
                    score = self._tone_window_energy(window, center_pos)
                if score > best_score:
                    best_score = score
                    best_symbol = m
            bits[i] = best_symbol

        return CSSRxResult(waveform=wf, bits=bits, config=cfg)

    def _equalize(self, wf: np.ndarray) -> np.ndarray:
        """Deconvolve the received waveform's room impulse response (MMSE).

        When :attr:`CSSConfig.eq_rir` is set, the transmitted chirp sequence
        has been linearly convolved with the RIR (reverberation smears each
        de-chirped tone across neighbouring FFT bins, raising BER).  Here the
        channel is deconvolved in the frequency domain with an MMSE (Wiener)
        filter, :math:`\\hat X = Y H^* / (|H|^2 + \\sigma^2)`, which suppresses
        reverberation tail energy the way :meth:`MFSKModem.demodulate` does for
        FSK.  With :attr:`eq_rir` unset the waveform is returned unchanged.

        Args:
            wf: Received (possibly reverberated) real waveform.

        Returns:
            1-D real waveform, deconvolved when :attr:`eq_rir` is configured.
        """
        cfg = self._config
        if cfg.eq_rir is None:
            return wf
        h = np.asarray(cfg.eq_rir, dtype=np.float64)
        if h.ndim != 1 or h.size < 2:
            return wf
        n = wf.size
        H = np.fft.fft(h, n=n)
        if not np.any(np.abs(H) > 0.0):
            return wf
        noise_floor = float(np.max(np.abs(H) ** 2)) * 1e-3
        Y = np.fft.fft(wf, n=n)
        X_hat = Y * np.conj(H) / (np.abs(H) ** 2 + noise_floor)
        return np.fft.ifft(X_hat).real

    @staticmethod
    def _tone_window_energy(window: np.ndarray, center: int) -> float:
        """Distance-weighted FFT energy around a candidate tone bin.

        Energy within the tolerance window is weighted so the centre bin
        dominates and bins far from it (leakage from adjacent chirps) barely
        count.  This mirrors :meth:`MFSKModem.demodulate`'s weighted-argmax.
        """
        w = np.abs(window).astype(np.float64)
        n = w.size
        if n == 0:
            return 0.0
        if n == 1 or center <= 0 or center >= n:
            return float(w.sum())
        dist = np.abs(np.arange(n) - center)
        weights = 1.0 / (1.0 + 2.0 * dist)
        return float(np.dot(w, weights))

    # -- BER helper (mirrors MFSKModem.ber) ----------------------------------

    def ber(
        self,
        transmitted: np.ndarray | Sequence[int],
        received: np.ndarray | Sequence[int],
    ) -> float:
        """Return the bit error rate for a transmitted vs. received pair.

        ``transmitted`` and ``received`` may be raw data bits (as produced by
        :meth:`encode`) or chirp indices (as produced by :meth:`demodulate`);
        either representation is normalized to a symbol-index stream before
        comparing, so the method works on both sides.  BER is
        ``bit_errors / total_bits`` with
        ``total_bits = num_common_symbols * bits_per_symbol``.

        Args:
            transmitted: Transmitted bits or chirp indices.
            received: Received bits or chirp indices.

        Returns:
            Bit error rate in ``[0, 1]`` (or ``nan`` if nothing is compared).
        """
        tx = np.asarray(transmitted)
        rx = np.asarray(received)
        if tx.ndim != 1 or rx.ndim != 1:
            raise ValueError("transmitted and received must be 1-D arrays")
        bps = self._config.bits_per_symbol
        n_chirps = self._config.n_chirps

        def _to_symbols(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(np.int64)
            if a.size % bps == 0 and a.max(initial=0) <= 1:
                return self.symbols_from_bits(a)
            return _as_valid_symbols(a, n_chirps)

        s_tx = _to_symbols(tx)
        s_rx = _to_symbols(rx)
        n = min(s_tx.size, s_rx.size)
        if n == 0:
            return float("nan")
        total_bits = n * bps
        bit_errors = int(np.sum(s_tx[:n] != s_rx[:n])) * bps
        return float(bit_errors) / float(total_bits)


__all__ = [
    "CSSConfig",
    "CSSModem",
    "CSSRxResult",
    "CSSTxOutput",
    "DEFAULT_FS",
    "DEFAULT_FS_HZ",
    "DEFAULT_N_CHIRPS",
    "DEFAULT_SYMBOLS_PER_SECOND",
    "DEFAULT_TOTAL_BANDWIDTH",
]
