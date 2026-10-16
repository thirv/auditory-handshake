"""Physical layer (PHY) for the handshake protocol.

S13.1 adds the MFSK modem (:mod:`auditory_handshake.phy.modem`): a 4-FSK
data symbol is one pure tone drawn from a pre-agreed constellation, so a
receiver only needs an FFT and an argmax to decode.  The modem is
self-contained (clean mono waveform in, bits out).

S13.2 adds the acoustic link model (:mod:`auditory_handshake.phy.channel`):
apply RIR / Doppler / noise to a waveform with :func:`apply_acoustic_channel`,
and measure the resulting BER/BLER round-trip through a modem with
:func:`measure_link`.

:mod:`auditory_handshake.phy.css` adds a Chirp Spread Spectrum (CSS) modem:
instead of a pure tone per symbol it transmits a short linear frequency-modulated
(LFM) chirp, so that processing gain and frequency diversity improve robustness
against narrowband interference and reverberation relative to the MFSK modem.
The :class:`~auditory_handshake.phy.css.CSSModem` mirrors the MFSK modem's
interface (``modulate`` / ``demodulate`` / ``ber`` / ``encode`` / ``decode``).

:mod:`auditory_handshake.phy.preamble` adds the acoustic preamble and
matched-filter detector (milestone M4): :func:`generate_preamble` builds the
Zadoff-Chu (CAZAC) or chirp sync waveform, :class:`PreambleDetector` runs the
energy-normalised matched-filter scan, and :func:`measure_detection` reports
the M4 metrics — detection latency (p50/p99) and false-alarm rate.
"""

from auditory_handshake.phy.channel import (
    LinkStats,
    apply_acoustic_channel,
    measure_link,
)
from auditory_handshake.phy.css import (
    CSSConfig,
    CSSModem,
    CSSRxResult,
    CSSTxOutput,
)
from auditory_handshake.phy.modem import (
    DEFAULT_FS,
    DEFAULT_FS_HZ,
    DEFAULT_SYMBOLS_PER_SECOND,
    MFSKConfig,
    MFSKModem,
    MFSKRxResult,
    MFSKTxOutput,
)
from auditory_handshake.phy.preamble import (
    PREAMBLE_WAVES,
    PreambleConfig,
    PreambleDetectionResult,
    PreambleDetectionStats,
    PreambleDetector,
    PreambleWaveform,
    generate_preamble,
    measure_detection,
)

__all__ = [
    "CSSConfig",
    "CSSModem",
    "CSSRxResult",
    "CSSTxOutput",
    "DEFAULT_FS",
    "DEFAULT_FS_HZ",
    "DEFAULT_SYMBOLS_PER_SECOND",
    "LinkStats",
    "MFSKConfig",
    "MFSKModem",
    "MFSKRxResult",
    "MFSKTxOutput",
    "PREAMBLE_WAVES",
    "PreambleConfig",
    "PreambleDetectionResult",
    "PreambleDetectionStats",
    "PreambleDetector",
    "PreambleWaveform",
    "apply_acoustic_channel",
    "generate_preamble",
    "measure_detection",
    "measure_link",
]
