"""Preamble detection latency and false-alarm rate across rooms and SNR (M4).

Two preamble waveform families (PLAN §4 candidates) are swept over the bundled
room presets at a fixed SNR. Two arms are measured per room/waveform cell:

* ``awgn``   — :func:`measure_detection` directly (white-Gaussian background
  noise): the reference operating point and the source of the M4 latency/FAR
  headline numbers.
* ``channel``— the preamble is transmitted through a pyroomacoustics RIR
  computed for the room (target RT60 = :data:`RT60_TARGET_S`, Sabine formula)
  with additive noise matched to the same SNR, then the matched-filter detector
  is applied — the realistic-room arm.

Detection probability, false-alarm rate, and latency p50/p99 are written to a
CSV table plus a ``metrics.json`` summary (``write_metrics``), and one example
received preamble per waveform is dumped as WAV for inspection.

Exit criterion (PLAN §4): p99 detection latency < 8 ms.
"""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Sequence

import numpy as np
from scipy import signal

from auditory_handshake.acoustics.channel import compute_rir
from auditory_handshake.acoustics.room import calculate_rt60
from auditory_handshake.config import all_room_presets
from auditory_handshake.metrics import TTCH_TTFB_TARGET_MS, compute_ttfb
from auditory_handshake.phy.preamble import (
    PREAMBLE_WAVES,
    PreambleConfig,
    PreambleDetector,
    generate_preamble,
    measure_detection,
)
from auditory_handshake.results import create_run, generate_audio, write_metrics

#: Target reverberation time (s) for the channel arm — mid-sized room.
RT60_TARGET_S: float = 0.5
#: RT60 closed-form equation (``sabine`` or ``eyring``).
RT60_FORMULA: str = "sabine"
#: Preamble SNR (dB) for every cell.
SNR_DB: float = 20.0
#: Detection trials per cell (signal + noise halves).
TRIALS: int = 200
#: Receive buffer (s) for the detection sweeps.
BUFFER_S: float = 0.1
#: Embed the preamble in [MIN, MAX]·(buffer − preamble) to avoid edge effects.
MIN_OFFSET_FRAC: float = 0.0
MAX_OFFSET_FRAC: float = 0.7

CSV_COLUMNS: Sequence[str] = (
    "room",
    "rt60_s",
    "room_volume_m3",
    "snr_db",
    "wave",
    "arm",
    "detection_probability",
    "false_alarm_rate",
    "latency_p50_ms",
    "latency_p99_ms",
    "seed",
)


def _cell_seed(room_index: int, wave_index: int, arm_index: int) -> int:
    """Derive a stable, per-cell seed from the global seed."""
    return 1_000_003 * room_index + 7_919 * wave_index + 104_729 * arm_index + 42


def _rx_len(cfg: PreambleConfig) -> int:
    return 4 * int(round(cfg.fs * BUFFER_S)) + cfg.length


def _channel(
    cfg: PreambleConfig,
    rir: np.ndarray,
    rng: np.random.Generator,
    conv_once: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Transmit the preamble through ``rir`` and add SNR-matched noise.

    Returns ``(rx, true_offset)`` where ``rx`` is the received buffer and
    ``true_offset`` the embedded preamble start sample. Noise sigma is chosen
    so the preamble energy-to-noise-power ratio at the receiver equals
    ``SNR_DB`` (SNR defined over the full preamble energy, matching
    :func:`measure_detection`).

    ``conv_once`` — the precomputed FFT convolution of the preamble with the
    RIR (``signal.fftconvolve(preamble, rir)``) — is reused across trials so
    each trial costs O(buffer) instead of O(buffer × RIR).
    """
    preamble = generate_preamble(cfg).waveform
    n_preamble = cfg.length
    if conv_once is None:
        conv_once = signal.fftconvolve(preamble, rir)
    rx_len = _rx_len(cfg)
    max_offset = rx_len - n_preamble
    lo = max(0, int(MIN_OFFSET_FRAC * max_offset))
    hi = max(lo, int(MAX_OFFSET_FRAC * max_offset))
    true_offset = int(rng.integers(lo, hi + 1))

    channel = np.zeros(rx_len, dtype=np.float64)
    take = min(conv_once.size, rx_len - true_offset)
    channel[true_offset : true_offset + take] = conv_once[:take]

    ref = channel[true_offset : true_offset + n_preamble]
    ref_power = float(np.mean(np.abs(ref) ** 2))
    if ref_power <= 0.0:
        raise ValueError("channel produced zero preamble energy — check RIR")
    sigma = (
        cfg.amplitude
        * float(np.linalg.norm(preamble))
        / math.sqrt(ref_power * 10.0 ** (SNR_DB / 10.0))
    )
    rx = channel + rng.standard_normal(rx_len) * sigma
    return rx, true_offset


def _run_channel(cfg: PreambleConfig, rir: np.ndarray, seed: int) -> dict:
    """Channel arm: fresh noisy receive buffer per trial (detect + FAR + latency).

    Mirrors the trial structure of :func:`measure_detection` (half signal, half
    noise-only) so the two arms report comparable statistics.
    """
    detector = PreambleDetector(cfg)
    n_preamble = cfg.length
    rng = np.random.default_rng(seed)
    conv_once = signal.fftconvolve(generate_preamble(cfg).waveform, rir)
    rx_len = _rx_len(cfg)
    # Noise-only sigma: same SNR reference measured on the RIR's onset window.
    ref = rir[:n_preamble]
    ref_power = float(np.mean(np.abs(ref) ** 2))
    if ref_power <= 0.0:
        ref_power = 1.0
    sigma_noise = (
        cfg.amplitude
        * math.sqrt(float(np.sum(np.abs(ref) ** 2)))
        / math.sqrt(10.0 ** (SNR_DB / 10.0))
    )

    n_signal = TRIALS // 2
    n_noise = TRIALS - TRIALS // 2
    n_detected = 0
    n_false_alarms = 0
    latencies_s: list[float] = []

    for _ in range(n_signal):
        rx, true_offset = _channel(cfg, rir, rng, conv_once)
        result = detector.detect(rx)
        if result.found and result.offset is not None:
            n_detected += 1
            latencies_s.append((result.offset - true_offset + n_preamble - 1) / cfg.fs)
    for _ in range(n_noise):
        noise = rng.standard_normal(rx_len) * sigma_noise
        if detector.detect(noise).found:
            n_false_alarms += 1

    p50 = float(np.percentile(latencies_s, 50)) if latencies_s else float("nan")
    p99 = float(np.percentile(latencies_s, 99)) if latencies_s else float("nan")
    return {
        "n_trials": TRIALS,
        "n_signal_trials": n_signal,
        "n_noise_trials": n_noise,
        "n_detected": n_detected,
        "n_false_alarms": n_false_alarms,
        "detection_probability": n_detected / n_signal,
        "false_alarm_rate": n_false_alarms / n_noise,
        "latency_p50_s": p50,
        "latency_p99_s": p99,
        "snr_db": SNR_DB,
        "seed": seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None:
        print(f"warning: {len(argv)} extra CLI arg(s) ignored by sim 04", file=sys.stderr)

    rooms = all_room_presets()
    config = {
        "waveforms": list(PREAMBLE_WAVES),
        "snr_db": SNR_DB,
        "trials": TRIALS,
        "buffer_s": BUFFER_S,
        "min_offset_frac": MIN_OFFSET_FRAC,
        "max_offset_frac": MAX_OFFSET_FRAC,
        "rt60_target_s": RT60_TARGET_S,
        "rt60_formula": RT60_FORMULA,
        "rooms": [room.name for room in rooms],
    }
    run = create_run("sim_04_preamble_detection", seed=0, config=config)

    rows: list[dict[str, float | str]] = []
    n_preamble = PreambleConfig().length
    for room_index, room in enumerate(rooms):
        dims = room.dimensions_m
        volume = float(dims[0]) * float(dims[1]) * float(dims[2])
        rt60 = calculate_rt60(room, formula=RT60_FORMULA)
        rir = compute_rir(room, RT60_TARGET_S, fs=48_000.0)
        print(
            f"[sim 04] room={room.name}  dims={dims}  V={volume:.2f} m^3  "
            f"RT60={rt60 * 1000:.1f} ms  RIR={rir.size} samples"
        )

        for wave_index, wave in enumerate(PREAMBLE_WAVES):
            cfg = PreambleConfig(wave=wave, fs=48_000.0)
            for arm_index, arm in enumerate(("awgn", "channel")):
                seed = _cell_seed(room_index, wave_index, arm_index)
                if arm == "awgn":
                    stats = measure_detection(
                        cfg,
                        snr_db=SNR_DB,
                        n_trials=TRIALS,
                        buffer_s=BUFFER_S,
                        min_offset_frac=MIN_OFFSET_FRAC,
                        max_offset_frac=MAX_OFFSET_FRAC,
                        seed=seed,
                    )
                    cell = {
                        "n_trials": stats.n_trials,
                        "detection_probability": stats.detection_probability,
                        "false_alarm_rate": stats.false_alarm_rate,
                        "latency_p50_s": stats.latency_p50_s,
                        "latency_p99_s": stats.latency_p99_s,
                        "seed": seed,
                    }
                else:
                    cell = _run_channel(cfg, rir, seed)

                p50 = cell["latency_p50_s"]
                p99 = cell["latency_p99_s"]
                row = {
                    "room": room.name,
                    "rt60_s": round(rt60, 6),
                    "room_volume_m3": round(volume, 4),
                    "snr_db": SNR_DB,
                    "wave": wave,
                    "arm": arm,
                    "detection_probability": round(cell["detection_probability"], 4),
                    "false_alarm_rate": round(cell["false_alarm_rate"], 4),
                    "latency_p50_ms": round(p50 * 1000.0, 3) if p50 == p50 else None,
                    "latency_p99_ms": round(p99 * 1000.0, 3) if p99 == p99 else None,
                    "seed": cell["seed"],
                }
                rows.append(row)
                print(
                    f"    {arm:7s} {wave:10s} Pd={row['detection_probability']:.3f} "
                    f"FAR={row['false_alarm_rate']:.4f} "
                    f"p50={row['latency_p50_ms']} ms p99={row['latency_p99_ms']} ms"
                )

    # Example received preamble per waveform (channel arm, first room) for a
    # listenable / inspectable artifact.
    example_paths: dict[str, str] = {}
    if rooms:
        rir0 = compute_rir(rooms[0], RT60_TARGET_S, fs=48_000.0)
        rng0 = np.random.default_rng(1234)
        for wave in PREAMBLE_WAVES:
            cfg = PreambleConfig(wave=wave, fs=48_000.0)
            rx, _ = _channel(cfg, rir0, rng0)
            path = generate_audio(run.dir / f"preamble_{wave}.wav", rx, 48_000)
            example_paths[f"example_{wave}"] = str(path)

    # CSV
    csv_path = run.tables_dir / "preamble_detection.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})
    print(f"[sim 04] wrote {csv_path}")

    # Summary stats for the metrics file
    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    # TTFB / M4 exit criterion over the reference ``awgn`` arm (the documented
    # source of the M4 latency headline numbers). Compute once per wave with
    # the same parameters/seed as that arm, then summarise from the reports.
    ttfb_reports = {
        wave: compute_ttfb(
            PreambleConfig(wave=wave, fs=48_000.0),
            snr_db=SNR_DB,
            n_trials=TRIALS,
            buffer_s=BUFFER_S,
            min_offset_frac=MIN_OFFSET_FRAC,
            max_offset_frac=MAX_OFFSET_FRAC,
            seed=_cell_seed(0, wave_index, 0),
        )
        for wave_index, wave in enumerate(PREAMBLE_WAVES)
    }
    metrics = {
        "n_cells": len(rows),
        "n_rooms": len(rooms),
        "snr_db": SNR_DB,
        "trials_per_cell": TRIALS,
        "rt60_target_s": RT60_TARGET_S,
        "summary": {
            "p50_floor_ms": round(n_preamble / 48_000.0 * 1000.0, 3),
            "m4_exit_p99_ms": TTCH_TTFB_TARGET_MS,
            "mean_detection_probability": round(_mean("detection_probability"), 4),
            "max_false_alarm_rate": round(max(r["false_alarm_rate"] for r in rows), 4),
            "mean_latency_p50_ms": round(_mean("latency_p50_ms"), 3),
            "mean_latency_p99_ms": round(_mean("latency_p99_ms"), 3),
        },
        # TTFB (time to first bit) — M4 exit criterion: p99 TTFB < target for
        # every wave in the reference ``awgn`` arm.
        "ttfb": {
            "target_p99_ms": TTCH_TTFB_TARGET_MS,
            "per_wave": {
                wave: {
                    "ttfb_p50_ms": round(report.ttfb_p50_ms, 3),
                    "ttfb_p99_ms": round(report.ttfb_p99_ms, 3),
                    "achieved": report.achieved,
                }
                for wave, report in ttfb_reports.items()
            },
            "m4_pass": all(report.achieved for report in ttfb_reports.values()),
        },
        "example_wavs": example_paths,
    }
    metrics_path = write_metrics(run, metrics)
    print(f"[sim 04] wrote {metrics_path}")

    print(
        f"[sim 04] done. {len(rows)} cells across {len(rooms)} rooms "
        f"({SNR_DB} dB, {TRIALS} trials)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
