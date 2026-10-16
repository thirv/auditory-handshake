"""Acoustic preambles and matched-filter preamble detection (milestone M4).

PLAN §4 names the two candidate sync waveforms as *candidates to be tested, not
assumed*: Zadoff-Chu / CAZAC preambles for sample-accurate sync (single
correlation peak ⇒ lowest detection latency and a cheap false-alarm analysis)
and chirps for low-SNR / Doppler robustness. This module ships both generators
plus the common matched-filter detector, and the M4 measurements the milestone
exists to produce: **detection latency (p50/p99)** and **false-alarm rate**.

Conventions (shared with ``modem.py`` / ``css.py``):

- ``PreambleConfig`` is a frozen dataclass; defaults are ``DEFAULT_*`` constants
  and every field is validated in ``__post_init__`` (no silent fallbacks).
- All waveforms are real, passband (baseband symbol modulated onto ``fc_hz``),
  amplitude-scaled to ``amplitude`` — ready to feed ``AcousticChannel`` or the
  modem sim links directly.
- The detector's matched-filter output is *energy-normalised*: a perfectly
  matched window scores 1.0 (Cauchy–Schwarz bound), so ``detect_threshold`` is
  a meaningful, signal-level-independent decision level and the false-alarm
  analysis stays a one-parameter sweep (PLAN §4: "cheap false-alarm analysis").

Detection latency is measured from the first received preamble sample to the
earliest sample index at which the detector can decide — i.e. the last sample
consumed by the matched-filter window at the detected offset — so its floor is
the unavoidable preamble tail ``length / fs`` (≈ 1.3 ms at 48 kHz, N = 61),
well inside the M4 exit criterion "p99 detect < 8 ms".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "DEFAULT_AMPLITUDE",
    "DEFAULT_BUFFER_S",
    "DEFAULT_CHIRP_BANDWIDTH",
    "DEFAULT_DETECT_THRESHOLD",
    "DEFAULT_FC_HZ",
    "DEFAULT_FS",
    "DEFAULT_LENGTH",
    "DEFAULT_SNR_DB",
    "PREAMBLE_WAVES",
    "PreambleConfig",
    "PreambleDetectionResult",
    "PreambleDetectionStats",
    "PreambleDetector",
    "PreambleWaveform",
    "generate_preamble",
    "measure_detection",
]

#: Supported preamble waveform families (PLAN §4 candidates).
PREAMBLE_WAVES: tuple[str, str] = ("zadoff-chu", "chirp")

#: Sampling rate shared with the modems (``modem.DEFAULT_FS``).
DEFAULT_FS: float = 48_000.0
#: Passband centre frequency — sits between the MFSK (8–12 kHz) and CSS
#: (10–20 kHz) bands, so the same preamble can head-frame either modem.
DEFAULT_FC_HZ: float = 8_000.0
#: Preamble length in samples. 61 is prime ⇒ Zadoff-Chu is a full root CAZAC
#: sequence (zero periodic autocorrelation at all non-zero lags).
DEFAULT_LENGTH: int = 61
DEFAULT_AMPLITUDE: float = 1.0
#: Chirp sweep bandwidth (Hz) — the LFM sweeps ``[fc - B/2, fc + B/2]``.
DEFAULT_CHIRP_BANDWIDTH: float = 2_000.0
#: Normalised matched-filter decision threshold (1.0 = perfect match).
DEFAULT_DETECT_THRESHOLD: float = 0.7
#: Receive-buffer window for the latency/FAR measurement (0.1 s @ 48 kHz).
DEFAULT_BUFFER_S: float = 0.1
#: Preamble SNR (dB) for the default measurement operating point.
DEFAULT_SNR_DB: float = 20.0


def _is_prime(n: int) -> bool:
    """True if ``n`` is prime (trial division — lengths are tens, not millions)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


@dataclass(frozen=True)
class PreambleConfig:
    """Preamble + detector configuration (all fields validated at construction).

    Raises
    ------
    ValueError
        If ``wave`` is not in ``PREAMBLE_WAVES``, the carrier or chirp band
        escapes the Nyquist limit, ``length`` is not prime for ``zadoff-chu``,
        or ``amplitude`` / ``detect_threshold`` are out of range.
    """

    wave: str = "zadoff-chu"
    fs: float = DEFAULT_FS
    fc_hz: float = DEFAULT_FC_HZ
    length: int = DEFAULT_LENGTH
    amplitude: float = DEFAULT_AMPLITUDE
    chirp_bandwidth: float = DEFAULT_CHIRP_BANDWIDTH
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD

    def __post_init__(self) -> None:
        if self.wave not in PREAMBLE_WAVES:
            raise ValueError(f"wave must be one of {PREAMBLE_WAVES!r}, got {self.wave!r}")
        nyq = self.fs / 2.0
        if self.fs < 1.0:
            raise ValueError(f"fs must be >= 1 Hz, got {self.fs}")
        if not (1.0 <= self.fc_hz < nyq):
            raise ValueError(f"fc_hz must be in [1, fs/2) = [1, {nyq:.1f}) Hz, got {self.fc_hz}")
        if self.length < 8:
            raise ValueError(f"length must be >= 8 samples, got {self.length}")
        if self.amplitude <= 0.0:
            raise ValueError(f"amplitude must be > 0, got {self.amplitude}")
        if not (0.0 < self.detect_threshold < 1.0):
            raise ValueError(
                "detect_threshold must be in (0, 1) — 1.0 is a perfect "
                f"match — got {self.detect_threshold}"
            )
        if self.wave == "zadoff-chu" and not _is_prime(self.length):
            raise ValueError(
                "zadoff-chu length must be prime for CAZAC properties; "
                f"got {self.length} (e.g. use 61 or 67)"
            )
        if self.wave == "chirp":
            if self.chirp_bandwidth < 1.0:
                raise ValueError(f"chirp_bandwidth must be >= 1 Hz, got {self.chirp_bandwidth}")
            if self.fc_hz + self.chirp_bandwidth / 2.0 > nyq:
                raise ValueError(
                    "chirp band [fc - B/2, fc + B/2] escapes the Nyquist limit "
                    f"({nyq:.1f} Hz): fc={self.fc_hz}, B={self.chirp_bandwidth}"
                )


@dataclass(frozen=True)
class PreambleWaveform:
    """Generated preamble samples plus the config that produced them."""

    waveform: np.ndarray
    config: PreambleConfig


def generate_preamble(
    config: PreambleConfig | None = None,
) -> PreambleWaveform:
    """Generate the passband preamble for ``config``.

    ``zadoff-chu``: real part of ``exp(j·2π·fc·t) · exp(-jπ·n(n+1)/N)``,
    the root CAZAC sequence at prime length ``N`` — unit-magnitude baseband,
    zero periodic autocorrelation at every non-zero lag.

    ``chirp``: linear-frequency-modulated sweep over
    ``[fc - B/2, fc + B/2]`` across ``length`` samples (quadratic phase).
    """
    cfg = config if config is not None else PreambleConfig()
    n = np.arange(cfg.length, dtype=np.float64)
    if cfg.wave == "zadoff-chu":
        baseband = np.exp(-1j * math.pi * n * (n + 1.0) / cfg.length)
    else:
        k = cfg.chirp_bandwidth / (2.0 * cfg.fs * (cfg.length - 1))
        baseband = np.exp(2j * math.pi * k * n * n)
    waveform = cfg.amplitude * np.cos(2.0 * math.pi * cfg.fc_hz / cfg.fs * n + np.angle(baseband))
    return PreambleWaveform(waveform=waveform, config=cfg)


@dataclass(frozen=True)
class PreambleDetectionResult:
    """Outcome of one ``PreambleDetector.detect`` call.

    ``offset`` is the sample index of the best preamble-start candidate
    (``None`` when the peak fell below ``threshold``); ``peak`` is the
    normalised matched-filter level at that candidate (1.0 = perfect match);
    ``envelope`` is the full normalised correlation scan (for FAR plots).
    """

    found: bool
    offset: int | None
    peak: float
    threshold: float
    envelope: np.ndarray


@dataclass(frozen=True)
class PreambleDetectionStats:
    """M4 measurement: detection latency (p50/p99) and false-alarm rate.

    ``detection_probability`` — fraction of preamble-embedded trials detected.
    ``false_alarm_rate`` — fraction of noise-only trials the detector fired on.
    Latencies are measured from the first received preamble sample to the last
    sample the matched-filter window had to consume to decide, so the floor is
    the preamble tail ``length / fs``.
    """

    n_trials: int
    n_signal_trials: int
    n_noise_trials: int
    n_detected: int
    n_false_alarms: int
    detection_probability: float
    false_alarm_rate: float
    latency_p50_s: float
    latency_p99_s: float
    snr_db: float
    seed: int
    config: PreambleConfig


class PreambleDetector:
    """Energy-normalised matched-filter preamble detector.

    The scan is ``|x ⊛ p| / ||p||²`` over all valid alignments: a perfectly
    matched window scores exactly 1.0 and everything else — noise, the wrong
    preamble, partial overlaps — scores below it, which is what makes the
    threshold a single operating parameter and the FAR a clean sweep
    (PLAN §4: "cheap false-alarm analysis").
    """

    __slots__ = ("_config", "_preamble")

    def __init__(self, config: PreambleConfig | None = None) -> None:
        self._config = config if config is not None else PreambleConfig()
        self._preamble: np.ndarray | None = None

    @property
    def config(self) -> PreambleConfig:
        return self._config

    @property
    def preamble(self) -> np.ndarray:
        """The reference preamble (generated once, lazily)."""
        if self._preamble is None:
            self._preamble = generate_preamble(self._config).waveform
        return self._preamble

    def matched_filter(self, waveform: ArrayLike) -> np.ndarray:
        """Normalised matched-filter envelope (one value per alignment).

        Raises
        ------
        ValueError
            If ``waveform`` is shorter than the preamble.
        """
        x = np.asarray(waveform, dtype=np.float64).ravel()
        p = self.preamble
        if x.size < p.size:
            raise ValueError(
                "waveform must be at least the preamble length " f"({p.size} samples), got {x.size}"
            )
        corr = np.correlate(x, p, mode="valid")
        return np.abs(corr) / float(np.dot(p, p))

    def detect(self, waveform: ArrayLike) -> PreambleDetectionResult:
        """Locate the preamble start in ``waveform`` (best peak above threshold)."""
        envelope = self.matched_filter(waveform)
        offset = int(np.argmax(envelope))
        peak = float(envelope[offset])
        threshold = self._config.detect_threshold
        found = peak >= threshold
        return PreambleDetectionResult(
            found=found,
            offset=offset if found else None,
            peak=peak,
            threshold=threshold,
            envelope=envelope,
        )


def measure_detection(
    config: PreambleConfig | None = None,
    *,
    snr_db: float = DEFAULT_SNR_DB,
    n_trials: int = 200,
    buffer_s: float = DEFAULT_BUFFER_S,
    min_offset_frac: float = 0.0,
    max_offset_frac: float = 0.7,
    seed: int = 0,
) -> PreambleDetectionStats:
    """M4 measurement: detection latency (p50/p99) and false-alarm rate.

    Half the trials embed the preamble in white-Gaussian background noise at a
    uniformly random offset (SNR defined over the full preamble energy); the
    other half are noise-only at the same noise level, and any firing is a
    false alarm. All randomness is driven by ``seed``.
    """
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2, got {n_trials}")
    if not (0.0 <= min_offset_frac <= max_offset_frac <= 1.0):
        raise ValueError(
            "require 0 <= min_offset_frac <= max_offset_frac <= 1; "
            f"got {min_offset_frac}, {max_offset_frac}"
        )
    cfg = config if config is not None else PreambleConfig()
    n = cfg.length
    detector = PreambleDetector(cfg)
    buffer_samples = max(n + 1, int(round(cfg.fs * buffer_s)))
    max_offset = buffer_samples - n
    lo = max(0, int(min_offset_frac * max_offset))
    hi = max(lo, int(max_offset_frac * max_offset))

    # Noise std so the embedded preamble sits at ``snr_db``: signal energy
    # E = ||p||², noise variance σ² ⇒ SNR = E / σ².
    signal_norm = float(np.linalg.norm(detector.preamble))
    sigma = cfg.amplitude * signal_norm / math.sqrt(10.0 ** (snr_db / 10.0))

    rng = np.random.default_rng(seed)
    n_signal = n_trials // 2
    n_noise = n_trials - n_signal
    n_detected = 0
    n_false_alarms = 0
    latencies_s: list[float] = []

    for i in range(n_trials):
        x = rng.standard_normal(buffer_samples) * sigma
        if i < n_signal:
            offset = int(rng.integers(lo, hi + 1))
            x[offset : offset + n] += detector.preamble
            result = detector.detect(x)
            if result.found and result.offset is not None:
                n_detected += 1
                # Latency from preamble arrival: time from the first embedded
                # sample to the last sample the matched-filter window consumes
                # at the detected offset. A perfect lock lands at the floor
                # (n - 1) / fs; a false lock at a wrong offset shows up as a
                # large penalty, so the p50/p99 stats stay meaningful.
                latencies_s.append((result.offset - offset + n - 1) / cfg.fs)
        else:
            if detector.detect(x).found:
                n_false_alarms += 1

    if latencies_s:
        p50 = float(np.percentile(latencies_s, 50))
        p99 = float(np.percentile(latencies_s, 99))
    else:
        p50 = float("nan")
        p99 = float("nan")
    return PreambleDetectionStats(
        n_trials=n_trials,
        n_signal_trials=n_signal,
        n_noise_trials=n_noise,
        n_detected=n_detected,
        n_false_alarms=n_false_alarms,
        detection_probability=n_detected / n_signal,
        false_alarm_rate=n_false_alarms / n_noise,
        latency_p50_s=p50,
        latency_p99_s=p99,
        snr_db=snr_db,
        seed=seed,
        config=cfg,
    )
