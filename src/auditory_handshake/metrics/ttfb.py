"""TTFB — time to first bit — of the acoustic preamble handshake (M4).

TTFB is the latency from the first received preamble sample to the detector's
commit decision (matched-filter peak above threshold). That is *exactly* the
detection latency measured by :func:`auditory_handshake.phy.preamble.measure_detection`
(``latency_p50_s`` / ``latency_p99_s``), so this module is a thin, pure
metrics layer over those stats — the same relationship as
:mod:`auditory_handshake.metrics.ttch`, which wraps the CSS codec's symbol
count without re-measuring it.

The headline numbers come from the AWGN arm at the reference SNR; the
M4 exit criterion (PLAN.md §4) is **p99 TTFB < 8 ms**, which is also the
upper bound of the design-target row "Acoustic preamble detect + header
parse" (6–8 ms, THIS_WORK) in
:mod:`auditory_handshake.models.rf_baseline`.

Deterministic for a fixed ``seed`` (all randomness lives in
``measure_detection``); no acoustics, no I/O. Run
``python -m auditory_handshake.metrics.ttfb`` for a quick self-check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from auditory_handshake.phy.preamble import (
    DEFAULT_BUFFER_S,
    DEFAULT_SNR_DB,
    PREAMBLE_WAVES,
    PreambleConfig,
    measure_detection,
)

__all__ = [
    "TTCH_TTFB_TARGET_MS",
    "TTFBReport",
    "compute_ttfb",
]

#: M4 exit criterion (PLAN §4): p99 TTFB must stay below this (ms).
TTCH_TTFB_TARGET_MS: float = 8.0


@dataclass(frozen=True)
class TTFBReport:
    """One TTFB measurement (a single room/wave/arm cell or an AWGN point).

    ``ttfb_p50_ms`` / ``ttfb_p99_ms`` are NaN when no trial was detected;
    ``achieved`` is then ``False`` (a NaN can never satisfy the target —
    absence of a detection is never a pass).
    """

    n_trials: int
    n_signal_trials: int
    n_noise_trials: int
    n_detected: int
    n_false_alarms: int
    detection_probability: float
    false_alarm_rate: float
    ttfb_p50_ms: float
    ttfb_p99_ms: float
    snr_db: float
    seed: int
    wave: str
    target_ms: float
    achieved: bool


def compute_ttfb(
    config: PreambleConfig | None = None,
    *,
    snr_db: float = DEFAULT_SNR_DB,
    n_trials: int = 200,
    buffer_s: float = DEFAULT_BUFFER_S,
    min_offset_frac: float = 0.0,
    max_offset_frac: float = 0.7,
    seed: int = 0,
    wave: str | None = None,
    target_ms: float = TTCH_TTFB_TARGET_MS,
) -> TTFBReport:
    """Measure TTFB (detection latency) and evaluate it against the M4 target.

    Thin wrapper over :func:`measure_detection`: the latency definition
    (first received sample → decision sample, floor ``length / fs``) lives
    there, so TTFB and the M4 latency stats can never drift apart.

    Parameters
    ----------
    config:
        Preamble configuration (defaults: first of
        :data:`PREAMBLE_WAVES`, 48 kHz).
    wave:
        Override ``config.wave`` when ``config`` is ``None``.
    target_ms:
        M4 exit threshold for p99 TTFB (default 8.0 ms, PLAN §4).

    Raises
    ------
    ValueError
        If ``n_trials`` / offset bounds are invalid (checked by
        ``measure_detection``).
    """
    cfg = config if config is not None else PreambleConfig()
    if wave is not None and cfg.wave != wave:
        cfg = replace(cfg, wave=wave)
    stats = measure_detection(
        cfg,
        snr_db=snr_db,
        n_trials=n_trials,
        buffer_s=buffer_s,
        min_offset_frac=min_offset_frac,
        max_offset_frac=max_offset_frac,
        seed=seed,
    )
    achieved = isfinite(stats.latency_p99_s) and stats.latency_p99_s * 1000.0 < target_ms
    return TTFBReport(
        n_trials=stats.n_trials,
        n_signal_trials=stats.n_signal_trials,
        n_noise_trials=stats.n_noise_trials,
        n_detected=stats.n_detected,
        n_false_alarms=stats.n_false_alarms,
        detection_probability=stats.detection_probability,
        false_alarm_rate=stats.false_alarm_rate,
        ttfb_p50_ms=stats.latency_p50_s * 1000.0,
        ttfb_p99_ms=stats.latency_p99_s * 1000.0,
        snr_db=stats.snr_db,
        seed=stats.seed,
        wave=cfg.wave,
        target_ms=target_ms,
        achieved=achieved,
    )


if __name__ == "__main__":
    for wave in PREAMBLE_WAVES:
        report = compute_ttfb(PreambleConfig(wave=wave), wave=wave)
        status = "PASS" if report.achieved else "FAIL"
        print(
            f"[ttfb] {status} wave={report.wave:10s} "
            f"p50={report.ttfb_p50_ms:.2f} ms p99={report.ttfb_p99_ms:.2f} ms "
            f"(target p99 < {report.target_ms:.1f} ms, "
            f"n_detected={report.n_detected}/{report.n_signal_trials}, "
            f"seed={report.seed})"
        )
