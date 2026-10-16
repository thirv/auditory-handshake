"""Map metric design points to concrete PHY configurations (PLAN §5, M3).

This module bridges the *metric* layer (S12: the goodput–audibility trade-off
surface in :mod:`auditory_handshake.metrics`) and the *physical* layer (S13:
the MFSK and CSS modems plus the acoustic channel in
:mod:`auditory_handshake.phy`).  A :class:`~auditory_handshake.metrics.rate.
DesignPoint` is a dimensionless description of a link — how many sub-bands,
how wide each is, at what frequencies, at what audibility level — while an
``MFSKConfig`` / ``CSSConfig`` is a concrete, buildable modem.  This module
translates between the two and closes the loop by *measuring* the BLER a
design point actually achieves over the acoustic channel, so the metric-layer
BER assumption can be validated against (or replaced with) a physical
measurement.

Mapping rules (derived from the modem constellations)
------------------------------------------------------
A design point's occupied frequencies are interpreted as a *uniform* tone
lattice: the lowest occupied frequency becomes the base frequency
(``fs_hz``) and the median occupied-band spacing becomes the per-band width
(``spacing``).  :attr:`DesignPoint.bandwidth_per_band_hz` is the spacing
fallback when the frequency list has fewer than two entries.

MFSK (fixed 4-tone constellation, 2 bits/symbol): tone ``k`` sits at
``fs_hz + k·R_s``; feasible iff ``fs_hz >= band_lo`` and
``fs_hz + 3·R_s <= band_hi``, with ``n_bands == 4``.

CSS (``n`` chirps, ``log2(n)`` bits/symbol; chirp ``m`` starts at
``fs_hz + m·S`` with sweep width ``n·S``): feasible iff
``fs_hz >= band_lo`` and ``fs_hz + (2n-1)·S <= band_hi``, with ``n`` a
power of two >= 2.

Band limits default to ``audio_bands_hz`` in ``configs/defaults.yaml``
(:func:`auditory_handshake.config.load_defaults`); callers may override them
for what-if studies.  M3 acceptance criterion (PLAN §5): every design point
used for a decision must be band-feasible *and* have a measured BLER.

BLER measurement
----------------
:func:`measure_design_bler` modulates a (seedable) random symbol stream with
the mapped modem, pushes it through
:func:`auditory_handshake.phy.channel.apply_acoustic_channel` (optional RIR,
optional AWGN at a target SNR in dB), demodulates, and compares.  For MFSK
both bit-level and symbol-level (block) error rates are reported; for CSS
the demodulator yields chirp indices, so the block-error rate doubles as the
reported bit rate.  The measured BER is folded back into the design point to
produce an *evidence-corrected* goodput, per PLAN §5.

All functions are pure and deterministic (for a given ``seed``); nothing here
mutates its inputs or holds state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from auditory_handshake.config import load_defaults
from auditory_handshake.metrics.rate import (
    DesignPoint,
    aggregate_audibility_cost_db,
    goodput_bps,
)
from auditory_handshake.phy.channel import apply_acoustic_channel
from auditory_handshake.phy.css import CSSConfig, CSSModem
from auditory_handshake.phy.modem import (
    BITS_PER_SYMBOL,
    NUM_TONES,
    MFSKConfig,
    MFSKModem,
)

__all__ = [
    "DesignBlerResult",
    "DesignInfeasibleError",
    "ModemMode",
    "band_feasible",
    "band_limits",
    "design_to_css",
    "design_to_mfsk",
    "measure_design_bler",
]


#: Which modem family a design point should be realised with.
ModemMode = Literal["mfsk", "css"]


class DesignInfeasibleError(ValueError):
    """Raised when a design point cannot be realised by the requested modem.

    The message states *which* constraint was violated (band floor, band
    ceiling, constellation size, …) so callers can log a usable reason.
    """


def band_limits(
    mode: ModemMode,
    presets_dir: str | None = None,
) -> tuple[float, float]:
    """TX/RX band limits ``(band_lo_hz, band_hi_hz)`` for a modem mode.

    Args:
        mode: ``"mfsk"`` or ``"css"``.
        presets_dir: Optional root of the config directory (defaults to the
            repository's ``configs/`` as resolved by
            :func:`auditory_handshake.config.load_defaults`).

    Returns:
        A ``(lo, hi)`` tuple in Hz with ``0 < lo < hi``.

    Raises:
        ValueError: If ``mode`` is not a known modem family.
        KeyError: If the defaults file has no ``audio_bands_hz[mode]`` entry.
    """
    if mode not in ("mfsk", "css"):
        raise ValueError(f"mode must be 'mfsk' or 'css', got {mode!r}")
    limits = load_defaults(presets_dir)["audio_bands_hz"][mode]
    lo, hi = float(limits[0]), float(limits[1])
    if not (math.isfinite(lo) and lo > 0.0):
        raise ValueError(f"audio_bands_hz[{mode}] lower bound must be > 0 Hz, got {lo}")
    if not (math.isfinite(hi) and hi > lo):
        raise ValueError(f"audio_bands_hz[{mode}] upper bound must be > {lo} Hz, got {hi}")
    return lo, hi


def _base_and_spacing(design: DesignPoint) -> tuple[float, float]:
    """Extract ``(base_frequency_hz, per_band_spacing_hz)`` from a design point.

    The base frequency is the lowest occupied band.  The spacing is the
    median of the positive gaps between sorted occupied bands, falling back
    to :attr:`DesignPoint.bandwidth_per_band_hz` for a single band.
    """
    freqs = sorted(design.occupied_frequencies_hz)
    base = float(freqs[0])
    if len(freqs) >= 2:
        gaps = np.asarray(np.diff(freqs), dtype=np.float64)
        gaps = gaps[gaps > 0.0]
        if gaps.size:
            spacing = float(np.median(gaps))
            if math.isfinite(spacing) and spacing > 0.0:
                return base, spacing
    return base, design.bandwidth_per_band_hz


def _top_edge_hz(design: DesignPoint, mode: ModemMode, base_hz: float, spacing_hz: float) -> float:
    """Highest frequency any constellation element (or chirp edge) reaches."""
    n = design.n_bands
    if mode == "mfsk":
        return base_hz + (n - 1) * spacing_hz
    # CSS chirp m starts at base + m·S and sweeps n·S (total_bandwidth = n·S).
    return base_hz + (n - 1) * spacing_hz + n * spacing_hz


def _check_constellation(design: DesignPoint, mode: ModemMode) -> None:
    """Raise :class:`DesignInfeasibleError` if the band count is not realisable."""
    n = design.n_bands
    if mode == "mfsk" and n != NUM_TONES:
        raise DesignInfeasibleError(
            f"MFSK is a fixed {NUM_TONES}-tone modem, but the design point "
            f"occupies {n} band(s); cannot map 1:1"
        )
    if mode == "css" and (n < 2 or (n & (n - 1)) != 0):
        raise DesignInfeasibleError(f"CSS chirp count must be a power of two >= 2, got n_bands={n}")


def _check_bands(
    design: DesignPoint,
    mode: ModemMode,
    base_hz: float,
    top_hz: float,
    band_lo: float,
    band_hi: float,
) -> None:
    """Raise :class:`DesignInfeasibleError` if the constellation misses the band."""
    if base_hz < band_lo:
        raise DesignInfeasibleError(
            f"base frequency {base_hz:.1f} Hz is below the {mode} band floor "
            f"{band_lo:.1f} Hz (occupied band must stay inside the band)"
        )
    if top_hz > band_hi:
        raise DesignInfeasibleError(
            f"highest {mode} element reaches {top_hz:.1f} Hz, above the {mode} "
            f"band ceiling {band_hi:.1f} Hz (fs + {mode.upper()}-span must fit the band)"
        )


def band_feasible(
    design: DesignPoint,
    band_lo: float,
    band_hi: float,
    *,
    mode: ModemMode = "mfsk",
) -> bool:
    """Whether a design point fits the ``[band_lo, band_hi]`` TX/RX band.

    Checks, for the requested modem family:

    1. the occupied band count is realisable (MFSK: exactly 4 tones; CSS:
       power of two ≥ 2),
    2. the lowest occupied frequency is at or above ``band_lo``,
    3. the highest element — top MFSK tone ``fs + (n−1)·S``, or the top CSS
       chirp edge ``fs + (2n−1)·S`` — is at or below ``band_hi``.

    Args:
        design: The design point to test.
        band_lo: Band floor in Hz (must be > 0).
        band_hi: Band ceiling in Hz (must be > ``band_lo``).
        mode: Modem family to realise the design point with.

    Returns:
        ``True`` if every constraint holds, ``False`` otherwise.

    Raises:
        ValueError: If ``band_lo`` / ``band_hi`` are not a valid band or
            ``mode`` is unknown.
    """
    if mode not in ("mfsk", "css"):
        raise ValueError(f"mode must be 'mfsk' or 'css', got {mode!r}")
    if not (math.isfinite(band_lo) and band_lo > 0.0):
        raise ValueError(f"band_lo must be > 0 Hz, got {band_lo}")
    if not (math.isfinite(band_hi) and band_hi > band_lo):
        raise ValueError(f"band_hi must be > band_lo ({band_lo}), got {band_hi}")

    base_hz, spacing_hz = _base_and_spacing(design)
    try:
        _check_constellation(design, mode)
    except DesignInfeasibleError:
        return False
    return base_hz >= band_lo and (_top_edge_hz(design, mode, base_hz, spacing_hz) <= band_hi)


def design_to_mfsk(
    design: DesignPoint,
    *,
    band_lo: float | None = None,
    band_hi: float | None = None,
    presets_dir: str | None = None,
) -> MFSKConfig:
    """Map a design point to a concrete 4-FSK :class:`MFSKConfig`.

    ``fs_hz`` is the lowest occupied frequency and ``symbols_per_second`` is
    the median occupied-band spacing (MFSK's tone lattice *is*
    ``fs_hz + k·R_s``).  The mapped config is validated against the band
    limits before being returned.

    Args:
        design: A :class:`DesignPoint` with ``n_bands == 4``.
        band_lo: Band floor override (Hz).  Defaults to the ``mfsk`` entry of
            ``audio_bands_hz`` in ``defaults.yaml``.
        band_hi: Band ceiling override (Hz), same source.
        presets_dir: Config root for the default band limits.

    Returns:
        A validated :class:`MFSKConfig` whose 4 tones all sit inside
        ``[band_lo, band_hi]``.

    Raises:
        DesignInfeasibleError: If the design point cannot be realised (band
            count, band floor/ceiling, or spacing too small).
    """
    if band_lo is None or band_hi is None:
        lo, hi = band_limits("mfsk", presets_dir)
        band_lo = lo if band_lo is None else band_lo
        band_hi = hi if band_hi is None else band_hi
    _check_constellation(design, "mfsk")

    base_hz, spacing_hz = _base_and_spacing(design)
    symbols_per_second = int(round(spacing_hz))
    if symbols_per_second < 1:
        raise DesignInfeasibleError(
            f"tone spacing {spacing_hz} Hz rounds to {symbols_per_second}; "
            "MFSK symbol rate must be >= 1 symbol/s"
        )

    try:
        config = MFSKConfig(fs_hz=base_hz, symbols_per_second=symbols_per_second)
    except ValueError as exc:
        raise DesignInfeasibleError(f"design point violates modem limits: {exc}") from exc

    top_hz = config.fs_hz + (NUM_TONES - 1) * symbols_per_second
    _check_bands(design, "mfsk", base_hz, top_hz, band_lo, band_hi)
    return config


def design_to_css(
    design: DesignPoint,
    *,
    band_lo: float | None = None,
    band_hi: float | None = None,
    presets_dir: str | None = None,
) -> CSSConfig:
    """Map a design point to a concrete LFM-chirp (CSS) :class:`CSSConfig`.

    The chirp constellation mirrors the design point's band lattice: chirp
    ``m`` starts at ``fs_hz + m·S`` with ``S`` the per-band spacing, and the
    sweep width of each chirp (``total_bandwidth``) is ``n_bands·S`` so that
    adjacent chirps tile the whole occupied band.

    Args:
        design: A :class:`DesignPoint` whose ``n_bands`` is a power of two
            ≥ 2.
        band_lo: Band floor override (Hz).  Defaults to the ``css`` entry of
            ``audio_bands_hz`` in ``defaults.yaml``.
        band_hi: Band ceiling override (Hz), same source.
        presets_dir: Config root for the default band limits.

    Returns:
        A validated :class:`CSSConfig` whose chirp edges all sit inside
        ``[band_lo, band_hi]``.

    Raises:
        DesignInfeasibleError: If the design point cannot be realised (chirp
            count, band floor/ceiling, or spacing too small).
    """
    if band_lo is None or band_hi is None:
        lo, hi = band_limits("css", presets_dir)
        band_lo = lo if band_lo is None else band_lo
        band_hi = hi if band_hi is None else band_hi
    _check_constellation(design, "css")

    base_hz, spacing_hz = _base_and_spacing(design)
    total_bandwidth = design.n_bands * spacing_hz
    if total_bandwidth < 1.0:
        raise DesignInfeasibleError(f"chirp sweep width {total_bandwidth} Hz must be >= 1 Hz")

    try:
        config = CSSConfig(
            fs_hz=base_hz,
            n_chirps=design.n_bands,
            total_bandwidth=total_bandwidth,
        )
    except ValueError as exc:
        raise DesignInfeasibleError(f"design point violates modem limits: {exc}") from exc

    top_hz = _top_edge_hz(design, "css", base_hz, config.chirp_spacing)
    _check_bands(design, "css", base_hz, top_hz, band_lo, band_hi)
    return config


def _awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean Gaussian noise giving signal-to-noise power ratio ``snr_db``."""
    if not math.isfinite(snr_db) or snr_db <= -100.0:
        raise ValueError(f"snr_db must be finite and > -100 dB, got {snr_db}")
    sig_pow = float(np.mean(np.square(signal)))
    if sig_pow <= 0.0:
        raise ValueError("cannot derive AWGN from an all-zero signal")
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0) + 1.0)
    return rng.normal(0.0, math.sqrt(noise_pow), size=signal.size)


@dataclass(frozen=True)
class DesignBlerResult:
    """Measured link performance of one design point through the acoustic channel.

    Attributes:
        design: The design point that was mapped and measured.
        mode: Modem family used (``"mfsk"`` or ``"css"``).
        snr_db: The target SNR (dB) of the AWGN applied, or ``None`` for a
            clean channel (optional RIR only).
        num_symbols: Number of symbols (blocks) transmitted.
        ber: Measured bit-error rate in ``0.0 .. 1.0``.  For CSS the
            demodulator yields chirp indices, so this equals :attr:`bler`
            (one decision unit per symbol).
        bler: Measured block (symbol) error rate in ``0.0 .. 1.0``.
        goodput_bps: Evidence-corrected goodput — the design point's
            :func:`~auditory_handshake.metrics.rate.goodput_bps` recomputed
            with :attr:`ber` substituted for its theoretical value.
        audibility_cost_db: Total audibility cost of the design point
            (energy-aggregated per-tone costs).
    """

    design: DesignPoint
    mode: ModemMode
    snr_db: float | None
    num_symbols: int
    ber: float
    bler: float
    goodput_bps: float
    audibility_cost_db: float

    def __post_init__(self) -> None:
        if self.mode not in ("mfsk", "css"):
            raise ValueError(f"mode must be 'mfsk' or 'css', got {self.mode!r}")
        if self.num_symbols < 1:
            raise ValueError(f"num_symbols must be >= 1, got {self.num_symbols}")
        if not (0.0 <= self.ber <= 1.0):
            raise ValueError(f"ber must be in [0, 1], got {self.ber}")
        if not (0.0 <= self.bler <= 1.0):
            raise ValueError(f"bler must be in [0, 1], got {self.bler}")
        if self.ber > self.bler + 1e-12:
            raise ValueError(f"ber ({self.ber}) must be <= bler ({self.bler})")
        if not (math.isfinite(self.goodput_bps) and self.goodput_bps >= 0.0):
            raise ValueError(f"goodput_bps must be >= 0 and finite, got {self.goodput_bps}")
        if not math.isfinite(self.audibility_cost_db):
            raise ValueError(f"audibility_cost_db must be finite, got {self.audibility_cost_db}")


def measure_design_bler(
    design: DesignPoint,
    *,
    mode: ModemMode = "mfsk",
    num_symbols: int = 256,
    rir: np.ndarray | None = None,
    snr_db: float | None = None,
    seed: int | None = None,
    band_lo: float | None = None,
    band_hi: float | None = None,
    presets_dir: str | None = None,
) -> DesignBlerResult:
    """Measure the BLER a design point achieves over the acoustic channel.

    The flow is ``map → modulate → channel → demodulate → compare``: the
    modem comes from :func:`design_to_mfsk` / :func:`design_to_css` (so an
    infeasible design point raises :class:`DesignInfeasibleError` before any
    simulation), and the channel is
    :func:`auditory_handshake.phy.channel.apply_acoustic_channel` with an
    optional RIR and optional AWGN at target SNR.

    Args:
        design: The design point to evaluate.
        mode: Modem family (``"mfsk"`` or ``"css"``).
        num_symbols: Number of symbols (blocks) to transmit (≥ 1).
        rir: Optional 1-D room impulse response to convolve in.
        snr_db: Optional target SNR (dB) — zero-mean Gaussian noise is
            generated so that ``P_signal / P_noise = 10 ** (snr_db / 10)``.
            ``None`` runs a clean channel (RIR only if given).
        seed: RNG seed for the payload and noise (deterministic runs).
        band_lo, band_hi: Band-limit overrides for the mapping (see
            :func:`design_to_mfsk` / :func:`design_to_css`).
        presets_dir: Config root for the default band limits.

    Returns:
        A :class:`DesignBlerResult` with the measured BER/BLER, the
        evidence-corrected goodput (measured BER substituted into the design
        point), and the design point's audibility cost.

    Raises:
        DesignInfeasibleError: If the design point cannot be mapped to the
            requested modem under the band limits.
        ValueError: If ``num_symbols``, ``snr_db``, or ``mode`` is invalid.
    """
    if num_symbols < 1:
        raise ValueError(f"num_symbols must be >= 1, got {num_symbols}")

    if mode == "mfsk":
        mfsk_config = design_to_mfsk(
            design, band_lo=band_lo, band_hi=band_hi, presets_dir=presets_dir
        )
        config: MFSKConfig | CSSConfig = mfsk_config
        modem: MFSKModem | CSSModem = MFSKModem(mfsk_config)
        const = NUM_TONES
    else:
        css_config = design_to_css(
            design, band_lo=band_lo, band_hi=band_hi, presets_dir=presets_dir
        )
        config = css_config
        modem = CSSModem(css_config)
        const = css_config.n_chirps

    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, const, size=num_symbols).astype(np.int64)

    if mode == "mfsk":
        tx_waveform = modem.modulate(MFSKModem.bits_from_symbols(symbols)).waveform
    else:
        tx_waveform = modem.modulate(symbols).waveform

    noise = None
    if snr_db is not None:
        noise = _awgn(np.asarray(tx_waveform, dtype=np.float64), snr_db, rng)
    rx_waveform = apply_acoustic_channel(tx_waveform, config.fs, rir=rir, noise=noise)

    if mode == "mfsk":
        mfsk_modem = modem
        assert isinstance(mfsk_modem, MFSKModem)
        # ``MFSKModem.demodulate(...).bits`` holds one tone index (symbol) per
        # symbol -- not a bit stream -- so it must be expanded with
        # ``bits_from_symbols`` before any bit-level comparison (mirrors the
        # chirp-index handling in the CSS branch below).  Compare the common
        # prefix in case the acoustic channel changes the sample count.
        rx_symbols = mfsk_modem.demodulate(rx_waveform).bits
        n = min(rx_symbols.size, symbols.size)
        tx_sym = symbols[:n]
        rx_sym = rx_symbols[:n]
        wrong_sym = tx_sym != rx_sym
        num_block_errors = int(np.sum(wrong_sym))
        tx_bits = MFSKModem.bits_from_symbols(tx_sym)
        rx_bits = MFSKModem.bits_from_symbols(rx_sym)
        num_bit_errors = int(np.sum(tx_bits != rx_bits))
        num_bits = int(n * BITS_PER_SYMBOL)
    else:
        css_modem = modem
        assert isinstance(css_modem, CSSModem)
        rx_symbols = css_modem.demodulate(rx_waveform).bits
        num_block_errors = int(np.sum(rx_symbols != symbols))
        num_bit_errors = num_block_errors
        num_bits = int(num_symbols)

    ber = num_bit_errors / num_bits
    bler = num_block_errors / num_symbols

    corrected = replace(design, ber=ber)
    return DesignBlerResult(
        design=design,
        mode=mode,
        snr_db=snr_db,
        num_symbols=num_symbols,
        ber=ber,
        bler=bler,
        goodput_bps=goodput_bps(corrected),
        audibility_cost_db=aggregate_audibility_cost_db(design),
    )
