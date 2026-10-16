"""S11.1 — Deterministic physical-noise synthesis for scenario noise sources.

PLAN §2.5: "Small-quadrotor rotor noise is 70–90 dB SPL at 1 m, concentrated at
the blade-passing frequency and its harmonics (hundreds of Hz) with energy
decaying into the kHz range, plus broadband turbulence. HVAC/pink noise and
speech add more."

This module renders the *waveform* of each :class:`~auditory_handshake.config.NoiseSource`
so the S11 channel-characterisation sim has a realistic, reproducible noise floor to
add to the received signal (noise-adaptive spectral notching / per-sub-band MCS,
PLAN §2.5). S6 (:mod:`~auditory_handshake.metrics.link_budget`) treats the same
sources as an incoherent dB-SPL floor (``noise_floor_db``); here each source is
synthesised at its own SPL and the *incoherent* sum of the waveforms matches that
floor — powers add, levels don't — so the two views agree (see
:func:`synthesize_noise_floor`).

Supported ``kind`` values (exactly the ones that ship in ``configs/scenarios``):

- ``broadband``  — flat-spectrum ambient ("quiet" room), uniform PSD;
- ``hvac_pink``  — pink noise, power ∝ 1/f (−3 dB/octave PSD slope);
- ``speech``     — voiced speech: glottal harmonic stack (1/harmonic) through
                   formant resonators (≈500 / 1.5 / 2.5 kHz) with a syllabic
                   amplitude envelope, plus a faint unvoiced (fricative) floor;
- ``rotor``      — blade-passing tone stack at ``center_hz`` and its harmonics
                   (amplitudes ∝ 1/k**1.4, energy decaying into the kHz) plus a
                   low-level broadband turbulence component.

Every generator is *deterministic* for a fixed ``(source, duration, fs, seed)``
and draws from a per-source ``numpy.random.Generator`` (no global RNG state),
following the S10.2 channel convention: the same inputs reproduce bit-identical
waveforms.

Conventions (kept aligned with :mod:`~auditory_handshake.metrics.link_budget`):

- ``level_db_spl`` is the source's SPL **as it reaches the receiver/mic** (its
  ``distance_m`` is the geometry that value refers to). It sets the signal's RMS
  pressure relative to the 20 µPa reference. ``distance_m`` is retained for
  round-trip fidelity and is *not* re-applied here, so the rendered level matches
  :func:`~auditory_handshake.metrics.link_budget.noise_floor_db` exactly.
- Output is a 1-D ``float64`` pressure waveform of ``round(duration * fs)``
  samples.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np

from auditory_handshake.config import NoiseSource

__all__ = [
    "SUPPORTED_KINDS",
    "REF_P_PA",
    "synthesize_noise",
    "synthesize_noise_floor",
]

#: Reference sound pressure for the SPL scale: 0 dB SPL = 20 µPa.
REF_P_PA: float = 20.0e-6

#: ``kind`` values :func:`synthesize_noise` can render (the ones in configs).
SUPPORTED_KINDS: frozenset[str] = frozenset({"broadband", "hvac_pink", "speech", "rotor"})

# Spectral-shaping constants (kept private).
_PINK_FLOOR_HZ = 20.0  # low-freq floor for the 1/sqrt(f) pink rolloff (bounds DC).
_SPEECH_FORMANTS_HZ = (500.0, 1500.0, 2500.0)
_SPEECH_FORMANT_WIDTH_HZ = 260.0
_SPEECH_HIGH_KNEE_HZ = 4500.0  # gentle roll-off past the speech band.
_ROTOR_HARMONIC_DECAY = 1.4  # tone-stack amplitude ∝ 1/k**p (energy into kHz).
_ROTOR_TURBULENCE_FRAC = 0.15  # broadband turbulence level vs. the tone stack.
_ROTOR_DEFAULT_BPF_HZ = 500.0  # fallback blade-passing frequency when unset.


# ---------------------------------------------------------------------------
# Determinism & scaling helpers
# ---------------------------------------------------------------------------
def _rng_for(source: NoiseSource, seed: int, salt: int) -> np.random.Generator:
    """A per-source RNG: deterministic and distinct for each (source, salt) pair.

    The scenario ``seed`` is mixed with a stable CRC of the source identity and
    with ``salt`` (the source's list position) so that (a) a fixed
    ``(source, salt, seed)`` always yields the same stream, (b) two sources —
    even identical ones — at different positions draw independent streams, and
    (c) nothing relies on global RNG state.
    """
    ident = (
        source.kind,
        round(source.level_db_spl, 9),
        round(source.distance_m, 9),
        source.center_hz,
    )
    crc = zlib.crc32(repr(ident).encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng([int(seed), int(salt), int(crc)])


def _target_rms(level_db_spl: float) -> float:
    """RMS pressure (Pa) for an SPL (dB re 20 µPa)."""
    return float(REF_P_PA * 10.0 ** (float(level_db_spl) / 20.0))


def _normalize_rms(signal: np.ndarray, rms: float) -> np.ndarray:
    """Scale ``signal`` in place to the target RMS pressure; zero if degenerate."""
    current = float(np.sqrt(np.mean(np.square(signal))))
    if not np.isfinite(current) or current <= 0.0:
        return np.zeros_like(signal)
    return signal * (rms / current)


def _validate_geometry(duration_seconds: float, fs: float) -> int:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds!r}")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be positive, got {fs!r}")
    n = int(round(duration_seconds * fs))
    if n < 2:
        raise ValueError("duration_seconds * fs must yield at least 2 samples")
    return n


# ---------------------------------------------------------------------------
# Spectral-shaping primitives
# ---------------------------------------------------------------------------
def _pink_noise(rng: np.random.Generator, n: int, fs: float) -> np.ndarray:
    """Pink noise (power ∝ 1/f → |H| ∝ 1/sqrt(f)), DC-bounded below 20 Hz."""
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    h = 1.0 / np.sqrt(np.maximum(f, _PINK_FLOOR_HZ))
    return np.fft.irfft(h * np.fft.rfft(white), n=n)


def _speech_formant_filter(n: int, fs: float) -> np.ndarray:
    """Magnitude response: Lorentzian formant peaks + a high-frequency roll-off."""
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    h = np.zeros_like(f)
    for fc in _SPEECH_FORMANTS_HZ:
        h += 1.0 / np.hypot((f - fc) / _SPEECH_FORMANT_WIDTH_HZ, 1.0)
    h *= 1.0 / np.hypot(f / _SPEECH_HIGH_KNEE_HZ, 1.0)
    return h


# ---------------------------------------------------------------------------
# Per-kind raw (unnormalised) generators
# ---------------------------------------------------------------------------
def _gen_broadband(rng: np.random.Generator, n: int, fs: float) -> np.ndarray:
    """Flat-spectrum ambient — white noise (uniform PSD)."""
    return rng.standard_normal(n)


def _gen_speech(rng: np.random.Generator, n: int, fs: float) -> np.ndarray:
    """Voiced speech: glottal harmonic stack × formants, syllabic AM + fricative."""
    t = np.arange(n) / fs
    nyquist = fs / 2.0
    f0 = float(rng.uniform(100.0, 180.0))
    max_h = max(1, int(np.floor(nyquist / f0)))
    glottal = np.zeros(n)
    for k in range(1, max_h + 1):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        glottal += (1.0 / k) * np.sin(2.0 * np.pi * k * f0 * t + phase)

    h = _speech_formant_filter(n, fs)
    voiced = np.fft.irfft(h * np.fft.rfft(glottal), n=n)

    rate = float(rng.uniform(3.0, 6.0))
    env_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * rate * t + env_phase)
    return voiced * envelope + 0.15 * rng.standard_normal(n)


def _gen_rotor(rng: np.random.Generator, n: int, fs: float, center_hz: float | None) -> np.ndarray:
    """Blade-passing tone stack (BPF + harmonics) + low-level broadband turbulence."""
    t = np.arange(n) / fs
    nyquist = fs / 2.0
    bpf = float(center_hz) if center_hz is not None else _ROTOR_DEFAULT_BPF_HZ
    if not (np.isfinite(bpf) and 0.0 < bpf < nyquist):
        raise ValueError(f"rotor blade-passing frequency must be in (0, {nyquist}) Hz, got {bpf!r}")

    n_harm = int(np.floor(nyquist / bpf))
    signal = np.zeros(n)
    for k in range(1, n_harm + 1):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = 1.0 / (k**_ROTOR_HARMONIC_DECAY)
        signal += amp * np.sin(2.0 * np.pi * k * bpf * t + phase)

    turbulence = _ROTOR_TURBULENCE_FRAC * _pink_noise(rng, n, fs)
    return signal + turbulence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def synthesize_noise(
    source: NoiseSource,
    duration_seconds: float,
    fs: float,
    seed: int = 42,
    salt: int = 0,
) -> np.ndarray:
    """Render the waveform of a single noise source at its SPL.

    Args:
        source: A validated :class:`~auditory_handshake.config.NoiseSource`.
        duration_seconds: Length of the returned waveform in seconds.
        fs: Sampling rate (Hz) of the returned waveform.
        seed: Deterministic seed (scenario baseline is 42); see module docstring.
        salt: An extra integer mixed into the per-source RNG stream. It is ``0``
            for a lone source and set to the list position by
            :func:`synthesize_noise_floor`, so two *identical* sources in one
            scenario draw independent streams and stay incoherent when summed.

    Returns:
        A 1-D ``float64`` pressure waveform of ``round(duration_seconds * fs)``
        samples whose RMS equals ``source.level_db_spl`` (dB re 20 µPa).

    Raises:
        ValueError: If ``source.kind`` is unsupported, the geometry is invalid,
            or a rotor has no in-band blade-passing frequency.
    """
    if source.kind not in SUPPORTED_KINDS:
        raise ValueError(
            f"unsupported noise kind {source.kind!r}; expected one of {sorted(SUPPORTED_KINDS)}"
        )

    n = _validate_geometry(duration_seconds, fs)
    rng = _rng_for(source, seed, salt)

    kind = source.kind
    if kind == "broadband":
        raw = _gen_broadband(rng, n, fs)
    elif kind == "hvac_pink":
        raw = _pink_noise(rng, n, fs)
    elif kind == "speech":
        raw = _gen_speech(rng, n, fs)
    else:  # kind == "rotor"
        raw = _gen_rotor(rng, n, fs, source.center_hz)

    return _normalize_rms(raw, _target_rms(source.level_db_spl))


def synthesize_noise_floor(
    sources: Sequence[NoiseSource],
    duration_seconds: float,
    fs: float,
    seed: int = 42,
) -> np.ndarray:
    """Render the combined incoherent noise floor of several sources.

    Each source is synthesised at its own SPL (see :func:`synthesize_noise`) and
    the waveforms are summed. Because the per-source RNG streams are independent,
    the sum is incoherent and its RMS SPL matches
    :func:`~auditory_handshake.metrics.link_budget.noise_floor_db` over the same
    source levels (powers add, levels don't).
    """
    n = _validate_geometry(duration_seconds, fs)
    if not sources:
        return np.zeros(n, dtype=np.float64)
    # Each source gets a distinct salt (its position) so identical sources stay
    # incoherent and the sum matches noise_floor_db even for duplicated sources.
    parts = (synthesize_noise(s, duration_seconds, fs, seed, salt=i) for i, s in enumerate(sources))
    total = sum(parts)
    return np.asarray(total, dtype=np.float64)
