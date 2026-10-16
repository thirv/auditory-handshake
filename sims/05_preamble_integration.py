"""S14 — PreambleDetector-assisted CSS demodulation vs the S13 (sim_03) baseline.

S13 demodulated a CSS frame that is already time-aligned at sample 0 — the
receiver *knew* when the frame starts. This simulation closes that gap: the
transmitted frame is now a ``preamble + CSS payload`` that arrives at an
**unknown onset** inside a noise-filled receive buffer (the realistic
condition), and the receiver must find it.

Per (room × SNR) cell the frame is sent through the *same* per-room RIR and
the *same* per-SNR noise scaling as ``sims/03_css_vs_mfsk_ab.py``, and three
arms are measured on the same 64-byte payload / RIR / SNR (fairness contract
carried over from S13):

* ``oracle``   — CSS sliced at the true *received* onset: transmit onset + the
  channel's direct-path delay + preamble length.  Upper bound (the receiver
  knows both the frame onset and the channel); reproduces the S13 ``css``
  baseline condition with the channel timing accounted for.
* ``detected`` — :class:`auditory_handshake.phy.preamble.PreambleDetector`
  locates the onset via its matched filter; the CSS payload is sliced at the
  detected onset.  A missed detection (peak < threshold) counts as a total
  decode failure (BER = 1 for that trial).
* ``blind``    — no preamble knowledge: the receiver assumes a fixed timing
  error of ``±BLIND_ERRORS_MS`` (random sign per trial) — how the same CSS
  modem fares with naive, uncorrected timing.

Outputs (PLAN §12 contract, written to ``results/sim_05/<UTC-ts>/``)
--------------------------------------------------------------------
- ``config.json``                    — inputs (via ``create_run``).
- ``tables/ber_vs_snr.csv``          — one row per (room, snr, arm).
- ``tables/detection.csv``           — Pd + offset-error stats per (room, snr).
- ``figs/ber_vs_snr_per_room.png``   — BER vs SNR, one panel per room.
- ``figs/alignment_offset_error.png``— mean onset offset error vs SNR per room.
- ``metrics.json``                   — summary (via ``write_metrics``).

Run:  ``python sims/05_preamble_integration.py``   (or ``make sim-05``)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from auditory_handshake.acoustics.channel import compute_rir
from auditory_handshake.acoustics.room import calculate_rt60
from auditory_handshake.config import all_room_presets
from auditory_handshake.phy import CSSModem
from auditory_handshake.phy.preamble import PreambleDetector, generate_preamble
from auditory_handshake.results import create_run, write_metrics

# ---- simulation parameters (aligned with sim_03) --------------------------- #
FS_HZ = 48_000.0  # sample rate (Hz) — project standard
PAYLOAD_BYTES = 64  # fixed message: 512 data bits
PAYLOAD_BITS = PAYLOAD_BYTES * 8

SNR_DBS = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
WORKING_SNR_DB = 10.0  # target operating point (matches sim_03)
TRIALS_PER_CELL = 100  # fresh noisy buffers per (room, snr, arm)

BUFFER_S = 0.6  # receive buffer length (s)
ONSET_MIN_FRAC = 0.10  # frame onset in [MIN, MAX]·(buffer − frame)
ONSET_MAX_FRAC = 0.35
BLIND_ERRORS_MS = (0.5, 1.0, 2.0)  # assumed timing error, ±, per blind trial

BER_CSV_COLUMNS: tuple[str, ...] = ("room", "rt60_s", "snr_db", "arm", "ber")
DET_CSV_COLUMNS: tuple[str, ...] = (
    "room",
    "rt60_s",
    "snr_db",
    "detection_probability",
    "mean_peak",
    "offset_err_mean_ms",
    "offset_err_std_ms",
    "offset_err_p99_ms",
)


def _embed_frame(
    frame_rir: np.ndarray,
    onset: int,
    rx_len: int,
    rng: np.random.Generator,
    snr_db: float,
) -> np.ndarray:
    """Embed the channel-echoed frame at ``onset`` in a noise buffer at SNR.

    Noise power is set from the frame's own clean energy (post-RIR), the same
    receiver-side SNR definition sim_03 uses.
    """
    ref_power = float(np.mean(np.abs(frame_rir) ** 2))
    if ref_power <= 0.0:
        ref_power = 1.0
    sigma = math.sqrt(ref_power / 10.0 ** (snr_db / 10.0))
    rx = rng.standard_normal(rx_len) * sigma
    rx[onset : onset + frame_rir.size] += frame_rir
    return rx


def _css_ber(
    css: CSSModem, rx: np.ndarray, css_start: int, tx_bits: np.ndarray, num_symbols: int
) -> float:
    """BER of the CSS modem on the payload slice starting at ``css_start``.

    Returns 1.0 (total failure) when the slice is empty/too short or lies
    before the end of the preamble — i.e. the payload window is not usable.
    """
    css_len = num_symbols * css.samples_per_symbol
    payload = rx[css_start : css_start + css_len]
    if css_start <= 0 or payload.size < css.samples_per_symbol:
        return 1.0
    rx_symbols = np.asarray(css.demodulate(payload).bits, dtype=np.int64)
    n = min(rx_symbols.size, num_symbols)
    if n == 0:
        return 1.0
    rx_bits = css.bits_from_symbols(rx_symbols[:n])
    tx = tx_bits[: rx_bits.size]
    return float(np.mean(tx != rx_bits)) if rx_bits.size else 1.0


def main() -> None:
    seed = 20260912
    css = CSSModem()
    detector = PreambleDetector()
    cfg = detector.config
    n_preamble = cfg.length
    n_payload = (PAYLOAD_BITS + css.bits_per_symbol - 1) // css.bits_per_symbol
    frame_len = n_preamble + n_payload * css.samples_per_symbol
    rx_len = int(BUFFER_S * FS_HZ)

    presets = all_room_presets()
    run = create_run(
        "sim_05",
        seed=seed,
        config={
            "fs_hz": FS_HZ,
            "payload_bytes": PAYLOAD_BYTES,
            "preamble_length": n_preamble,
            "preamble_wave": cfg.wave,
            "detect_threshold": cfg.detect_threshold,
            "snr_dbs": [float(x) for x in SNR_DBS.tolist()],
            "trials_per_cell": TRIALS_PER_CELL,
            "buffer_s": BUFFER_S,
            "onset_frac": [ONSET_MIN_FRAC, ONSET_MAX_FRAC],
            "blind_errors_ms": list(BLIND_ERRORS_MS),
            "rooms": [p.name for p in presets],
            "arms": ["oracle", "detected", "blind"],
        },
    )

    ber_rows: list[dict[str, Any]] = []
    det_rows: list[dict[str, Any]] = []

    for preset in presets:
        rt60 = calculate_rt60(preset, formula="sabine", fs=FS_HZ)
        rir = compute_rir(preset, rt60, fs=FS_HZ)
        rng_cell = np.random.default_rng(seed)
        payload = rng_cell.bytes(PAYLOAD_BYTES)
        tx_symbols = css.encode(payload, n_payload)
        tx_bits = np.asarray(
            css.bits_from_symbols(np.asarray(tx_symbols, dtype=np.int64)),
            dtype=np.int64,
        )
        frame = np.concatenate([generate_preamble(cfg).waveform, css.modulate(tx_symbols).waveform])
        # Frame-anchored channel (matches sim_03's _channel): keep the direct +
        # within-frame echoes but drop the reverberant tail that would spill past
        # the frame end, so the echoed frame stays the same length as the clean
        # frame (frame_len) and fits the fixed receive buffer at any onset.
        frame_rir = signal.fftconvolve(frame, rir)[: frame.size]
        max_onset = int((1 - ONSET_MAX_FRAC) * (rx_len - frame_len))

        # Channel direct-path delay (samples).  ``compute_rir`` normalises the
        # direct path to unit peak but leaves it at the physical source->mic
        # propagation index (the RIR's dominant tap), not at sample 0: the
        # source/mic are offset by ``_SRC_MIC_OFFSET_M`` (see channel.py), so
        # the frame's CSS payload arrives ``onset + direct_path_delay`` samples
        # after its transmit onset.  The oracle knows the channel, so it slices
        # at that *received* CSS start -- the same base the detected arm
        # recovers via peak detection (``result.offset``).  Omitting this
        # offset mis-slices the payload by the whole direct-path delay and
        # collapses the oracle to random BER, which is why it must be included.
        direct_path_delay = int(np.argmax(np.abs(np.asarray(rir))))

        print(
            f"[sim 05] room={preset.name}  RT60={rt60 * 1000:.1f} ms  "
            f"frame={frame.size} samples ({frame.size / FS_HZ * 1000:.1f} ms)"
        )

        for snr_db in SNR_DBS:
            snr = float(snr_db)
            bers = {arm: [] for arm in ("oracle", "detected", "blind")}
            offsets: list[int] = []
            peaks: list[float] = []
            n_found = 0

            for trial in range(TRIALS_PER_CELL):
                rng = np.random.default_rng(
                    hash((preset.name, float(snr), trial, seed)) & 0xFFFFFFFF
                )
                onset = int(rng.integers(int(ONSET_MIN_FRAC * (rx_len - frame_len)), max_onset))
                rx = _embed_frame(frame_rir, onset, rx_len, rng, snr)
                frame_onset = onset
                # Oracle: the receiver knows the channel, so the CSS payload
                # starts at transmit onset + the channel's direct-path delay
                # (``direct_path_delay``), not the naive transmit-onset
                # assumption the blind arm uses below.
                css_onset = onset + direct_path_delay + n_preamble

                # --- oracle: slice at the true *received* CSS onset (upper) --
                bers["oracle"].append(_css_ber(css, rx, css_onset, tx_bits, n_payload))

                # --- detected: PreambleDetector locates the onset -----------
                result = detector.detect(rx)
                if result.found and result.offset is not None:
                    n_found += 1
                    offsets.append(result.offset - frame_onset)
                    peaks.append(result.peak)
                    bers["detected"].append(
                        _css_ber(css, rx, result.offset + n_preamble, tx_bits, n_payload)
                    )
                else:
                    peaks.append(result.peak)
                    bers["detected"].append(1.0)

                # --- blind: fixed uncorrected timing error, ±random sign ----
                err_ms = float(rng.choice(BLIND_ERRORS_MS)) * (1 if rng.random() < 0.5 else -1)
                blind_onset = frame_onset + int(round(err_ms * FS_HZ / 1000.0))
                bers["blind"].append(
                    _css_ber(css, rx, blind_onset + n_preamble, tx_bits, n_payload)
                )

            base = {
                "room": preset.name,
                "rt60_s": float(round(rt60, 6)),
                "snr_db": snr,
            }
            for arm in ("oracle", "detected", "blind"):
                ber_rows.append({**base, "arm": arm, "ber": float(np.mean(bers[arm]))})

            off = np.asarray(offsets, dtype=np.float64)
            det_rows.append(
                {
                    **base,
                    "detection_probability": round(n_found / TRIALS_PER_CELL, 4),
                    "mean_peak": round(float(np.mean(peaks)), 4) if peaks else None,
                    "offset_err_mean_ms": (
                        round(float(np.mean(off)) * 1000.0 / FS_HZ, 6) if off.size else None
                    ),
                    "offset_err_std_ms": (
                        round(float(np.std(off)) * 1000.0 / FS_HZ, 6) if off.size else None
                    ),
                    "offset_err_p99_ms": (
                        round(float(np.percentile(np.abs(off), 99)) * 1000.0 / FS_HZ, 6)
                        if off.size
                        else None
                    ),
                }
            )
            off_str = f"off_mean={float(np.mean(off)):+.1f} samp" if off.size else "off_mean=n/a"
            print(
                f"    snr={snr:5.1f} dB  Pd={n_found / TRIALS_PER_CELL:.2f}  "
                f"BER oracle={np.mean(bers['oracle']):.4g} "
                f"detected={np.mean(bers['detected']):.4g} "
                f"blind={np.mean(bers['blind']):.4g}  {off_str}"
            )

    ber_df = pd.DataFrame(ber_rows, columns=BER_CSV_COLUMNS)
    det_df = pd.DataFrame(det_rows, columns=DET_CSV_COLUMNS)
    ber_csv = run.tables_dir / "ber_vs_snr.csv"
    ber_df.to_csv(ber_csv, index=False)
    det_csv = run.tables_dir / "detection.csv"
    det_df.to_csv(det_csv, index=False)
    print(f"TABLE     : {ber_csv}")
    print(f"TABLE     : {det_csv}")

    _plot_ber(ber_df, run.figs_dir)
    _plot_alignment(det_df, run.figs_dir)

    metrics = _summarise(ber_df, det_df)
    metrics_path = write_metrics(run, metrics)
    print(f"METRICS   : {metrics_path}")

    print()
    print("S14 — PreambleDetector-assisted CSS vs S13 baseline (same RIR/SNR)")
    piv = ber_df.pivot_table(index="snr_db", columns="arm", values="ber")
    print(piv.round(6).to_string())
    print(
        f"\nAt SNR={WORKING_SNR_DB:.0f} dB (mean over rooms): "
        f"oracle={metrics['mean_ber_at_working_snr']['oracle']:.4g}  "
        f"detected={metrics['mean_ber_at_working_snr']['detected']:.4g}  "
        f"blind={metrics['mean_ber_at_working_snr']['blind']:.4g}"
    )


def _plot_ber(ber_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rooms = sorted(ber_df["room"].unique())
    ncols = 3
    nrows = math.ceil(len(rooms) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
    for ax, room in zip(axes.ravel(), rooms, strict=False):
        sub = ber_df[ber_df["room"] == room]
        for arm, color, marker in (
            ("oracle", "tab:green", "o"),
            ("detected", "tab:red", "s"),
            ("blind", "tab:gray", "^"),
        ):
            s = sub[sub["arm"] == arm]
            ax.semilogy(
                s["snr_db"],
                s["ber"].clip(lower=1e-6),
                marker + "-",
                color=color,
                lw=2.0,
                label=arm,
            )
        ax.axhline(1e-3, color="tab:gray", lw=1.0, ls=":", alpha=0.8)
        ax.set_title(room, fontsize=10)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes.ravel()[len(rooms) :]:
        ax.axis("off")
    fig.suptitle("S14 — CSS BER vs SNR with unknown arrival (per room)", y=1.02)
    fig.tight_layout()
    png = out_dir / "ber_vs_snr_per_room.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG       : {png}")


def _plot_alignment(det_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    room_snr = det_df[det_df["snr_db"] == WORKING_SNR_DB].groupby("room")
    rooms = [r for r in sorted(det_df["room"].unique()) if r in room_snr.groups]
    vals = [float(room_snr.get_group(r)["offset_err_mean_ms"].iloc[0]) for r in rooms]
    p99s = [float(room_snr.get_group(r)["offset_err_p99_ms"].iloc[0]) for r in rooms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    xpos = np.arange(len(rooms))
    ax1.bar(xpos, vals, color="tab:red", alpha=0.85)
    for x, v in zip(xpos, vals, strict=False):
        ax1.annotate(f"{v:+.2f} ms", (x, v), ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(rooms, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("mean onset offset error (ms)")
    ax1.set_title(f"PreambleDetector onset error @ SNR={WORKING_SNR_DB:.0f} dB")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(xpos, p99s, color="tab:orange", alpha=0.85)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(rooms, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("p99 |offset error| (ms)")
    ax2.set_title("p99 onset offset error")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png = out_dir / "alignment_offset_error.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG       : {png}")


def _summarise(ber_df: pd.DataFrame, det_df: pd.DataFrame) -> dict[str, Any]:
    working = ber_df[ber_df["snr_db"] == WORKING_SNR_DB].groupby("arm")["ber"].mean()
    return {
        "n_rooms": int(ber_df["room"].nunique()),
        "n_snr_points": int(ber_df["snr_db"].nunique()),
        "payload_bits": int(PAYLOAD_BITS),
        "trials_per_cell": TRIALS_PER_CELL,
        "working_snr_db": WORKING_SNR_DB,
        "mean_ber_at_working_snr": {
            "oracle": float(working.get("oracle", float("nan"))),
            "detected": float(working.get("detected", float("nan"))),
            "blind": float(working.get("blind", float("nan"))),
        },
        "mean_detection_probability": float(det_df["detection_probability"].mean()),
        "min_detection_probability": float(det_df["detection_probability"].min()),
        "mean_offset_err_abs_ms": float(det_df["offset_err_mean_ms"].abs().mean()),
        "max_offset_err_p99_ms": float(det_df["offset_err_p99_ms"].max()),
    }


if __name__ == "__main__":
    main()
