"""S13 — CSS vs MFSK A/B comparison over the acoustic channel (PLAN §4, §11, §12).

Standalone simulation (not imported by the test suite). It pits the two
candidate PHY modems — the CSS modem (:class:`auditory_handshake.phy.css.CSSModem`)
and the legacy 4-FSK modem (:class:`auditory_handshake.phy.MFSKModem`) — against
each other under **identical** acoustic conditions, to answer "which PHY copes
better with the room?"

Fairness contract (the whole point of an A/B test)
--------------------------------------------------
- **Same payload**: one fixed random 512-bit message for every trial.
- **Same channel**: for each bundled room preset the *deterministic* RIR is
  synthesised once at the room's Sabine RT60 and reused for both modems.
- **Same SNR**: additive noise is scaled so the receiver-side signal-to-noise
  ratio (post-RIR clean power ÷ noise power) equals the swept target, computed
  *per modem* from that modem's own received signal, so both are judged at the
  same SNR on the same channel.

For every (room × SNR) point the link is run as
``encode → modulate → RIR → noise → demodulate`` and BER is measured on the
decoded bits.

Outputs (PLAN §12 contract, written to ``results/sim_03/<UTC-ts>/``)
--------------------------------------------------------------------
- ``config.json``                   — inputs + environment (via ``create_run``).
- ``tables/ab_ber_vs_snr.csv``      — one row per (room, snr, modem).
- ``figs/css_vs_mfsk_per_room.png`` — BER vs SNR, one panel per room.
- ``figs/css_vs_mfsk_summary.png``  — BER vs SNR, all rooms (mean envelope).
- ``metrics.json``                  — summary metrics (via ``write_metrics``).
- A human-readable summary on stdout.

Run:  ``python sims/03_css_vs_mfsk_ab.py``   (or ``make sim-03``)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auditory_handshake.acoustics.channel import compute_rir
from auditory_handshake.acoustics.room import calculate_rt60
from auditory_handshake.config import all_room_presets
from auditory_handshake.phy import CSSConfig, CSSModem, MFSKConfig, MFSKModem
from auditory_handshake.results import create_run, write_metrics

# ---- simulation parameters ------------------------------------------------- #
FS_HZ = 48_000.0  # sample rate (Hz) — project standard
PAYLOAD_BYTES = 64  # fixed message: 512 data bits
PAYLOAD_BITS = PAYLOAD_BYTES * 8

SNR_DBS = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])

#: SNR at which we ask "does the modem actually work?" (target BER 1e-3).
WORKING_SNR_DB = 10.0


def _channel(
    waveform: np.ndarray, rir: np.ndarray, rng: np.random.Generator, snr_db: float
) -> np.ndarray:
    """Apply RIR + SNR-matched additive noise to *waveform* (fair per-modem SNR).

    The noise level is derived from *this* modem's own post-RIR clean power so
    that the swept SNR is identical for CSS and MFSK regardless of their
    (different) signal amplitudes.
    """
    clean = np.convolve(waveform, rir, mode="full")[: waveform.size]
    ref_power = float(np.mean(np.abs(clean) ** 2))
    if ref_power <= 0.0:
        return clean
    sigma = math.sqrt(ref_power / (10.0 ** (snr_db / 10.0)))
    noise = rng.normal(0.0, sigma, size=waveform.size)
    return clean + noise


def _run_mfsk(payload: bytes, rir: np.ndarray, rng: np.random.Generator, snr_db: float) -> float:
    """BER of the 4-FSK modem for one payload / RIR / SNR."""
    mfsk = MFSKModem(MFSKConfig(eq_rir=rir))
    bits_per_symbol = 2  # 4-FSK carries 2 data bits per symbol
    num_symbols = (PAYLOAD_BITS + bits_per_symbol - 1) // bits_per_symbol

    tx_symbols = MFSKModem.encode(payload, num_symbols)
    tx_bits = MFSKModem.bits_from_symbols(np.asarray(tx_symbols, dtype=np.int64))
    waveform = mfsk.modulate(tx_bits).waveform

    rx_waveform = _channel(waveform, rir, rng, snr_db)
    rx_symbols = mfsk.demodulate(rx_waveform).bits
    n = min(rx_symbols.size, num_symbols)
    rx_bits = MFSKModem.bits_from_symbols(np.asarray(rx_symbols[:n], dtype=np.int64))
    tx = np.asarray(tx_bits, dtype=np.int64)[: rx_bits.size]
    return float(np.mean(tx != rx_bits)) if rx_bits.size else 0.0


def _run_css(payload: bytes, rir: np.ndarray, rng: np.random.Generator, snr_db: float) -> float:
    """BER of the CSS modem for one payload / RIR / SNR.

    CSS is symmetric with MFSK: both carry ``bits_per_symbol = 2`` data bits
    per symbol (4-chirp / 4-FSK), so the same payload → symbols → modulate →
    demodulate → symbols → bits round-trip applies and BER is computed on bits.
    """
    css = CSSModem(CSSConfig(eq_rir=rir))
    bps = css.bits_per_symbol  # 2
    num_symbols = (PAYLOAD_BITS + bps - 1) // bps

    tx_symbols = css.encode(payload, num_symbols)
    tx_bits = css.bits_from_symbols(np.asarray(tx_symbols, dtype=np.int64))
    waveform = css.modulate(tx_symbols).waveform

    rx_waveform = _channel(waveform, rir, rng, snr_db)
    rx_symbols = np.asarray(css.demodulate(rx_waveform).bits, dtype=np.int64)
    n = min(rx_symbols.size, num_symbols)
    rx_bits = css.bits_from_symbols(rx_symbols[:n])
    tx = np.asarray(tx_bits, dtype=np.int64)[: rx_bits.size]
    return float(np.mean(tx != rx_bits)) if rx_bits.size else 0.0


def run_ab(seed: int) -> pd.DataFrame:
    """Sweep every (room × SNR) point for both modems; return a long table."""
    rows: list[dict[str, Any]] = []
    presets = all_room_presets()
    for preset in presets:
        rt60 = calculate_rt60(preset, formula="sabine", fs=FS_HZ)
        rir = compute_rir(preset, rt60, fs=FS_HZ)
        rng = np.random.default_rng(seed)
        payload = rng.bytes(PAYLOAD_BYTES)
        for snr_db in SNR_DBS:
            ber_mfsk = _run_mfsk(payload, rir, np.random.default_rng(seed), float(snr_db))
            ber_css = _run_css(payload, rir, np.random.default_rng(seed), float(snr_db))
            base = {"room": preset.name, "snr_db": float(snr_db), "rt60_s": float(rt60)}
            rows.append({**base, "modem": "mfsk", "ber": ber_mfsk})
            rows.append({**base, "modem": "css", "ber": ber_css})
    df = pd.DataFrame(rows)
    df = df.sort_values(["room", "snr_db", "modem"]).reset_index(drop=True)
    return df


def _plot_per_room(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rooms = sorted(df["room"].unique())
    n = len(rooms)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
    for ax, room in zip(axes.ravel(), rooms, strict=False):
        sub = df[df["room"] == room]
        for modem, color, marker in (("mfsk", "tab:blue", "o"), ("css", "tab:red", "s")):
            s = sub[sub["modem"] == modem]
            ax.semilogy(
                s["snr_db"],
                s["ber"].clip(lower=1e-6),
                marker + "-",
                color=color,
                lw=2.0,
                label=modem.upper(),
            )
        ax.axhline(1e-3, color="tab:gray", lw=1.0, ls=":", alpha=0.8)
        ax.set_title(room, fontsize=10)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("S13 — CSS vs MFSK: BER vs SNR per room", y=1.02)
    fig.tight_layout()
    png = out_dir / "css_vs_mfsk_per_room.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG       : {png}")


def _plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for modem, color, marker in (("mfsk", "tab:blue", "o"), ("css", "tab:red", "s")):
        sub = df[df["modem"] == modem].groupby("snr_db")["ber"].mean()
        ax.semilogy(
            sub.index,
            np.clip(sub.values, 1e-6, None),
            marker + "-",
            color=color,
            lw=2.0,
            label=modem.upper(),
        )
    ax.axhline(1e-3, color="tab:gray", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER (mean over rooms)")
    ax.set_title(
        f"S13 — CSS vs MFSK: mean BER vs SNR  ({len(df['room'].unique())} rooms, "
        f"{PAYLOAD_BITS} bits/msg)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    png = out_dir / "css_vs_mfsk_summary.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"PNG       : {png}")


def main() -> None:
    seed = 20260912
    run = create_run(
        "sim_03",
        seed=seed,
        config={
            "fs_hz": FS_HZ,
            "payload_bytes": PAYLOAD_BYTES,
            "snr_dbs": [float(x) for x in SNR_DBS.tolist()],
            "rooms": [p.name for p in all_room_presets()],
            "modems": ["mfsk", "css"],
        },
    )

    df = run_ab(seed)
    csv = run.tables_dir / "ab_ber_vs_snr.csv"
    df.to_csv(csv, index=False)
    print(f"TABLE     : {csv}")

    _plot_per_room(df, run.figs_dir)
    _plot_summary(df, run.figs_dir)

    working = df[df["snr_db"] == WORKING_SNR_DB].groupby("modem")["ber"].mean()
    mfsk_w = float(working.get("mfsk", float("nan")))
    css_w = float(working.get("css", float("nan")))
    winner = "css" if css_w <= mfsk_w else "mfsk"

    metrics = {
        "n_rooms": int(df["room"].nunique()),
        "n_snr_points": int(df["snr_db"].nunique()),
        "payload_bits": int(PAYLOAD_BITS),
        "working_snr_db": WORKING_SNR_DB,
        "mean_ber_at_working_snr": {"mfsk": mfsk_w, "css": css_w},
        "lower_ber_at_working_snr": winner,
        "ber_range": {
            "mfsk_min": float(df.loc[df["modem"] == "mfsk", "ber"].min()),
            "mfsk_max": float(df.loc[df["modem"] == "mfsk", "ber"].max()),
            "css_min": float(df.loc[df["modem"] == "css", "ber"].min()),
            "css_max": float(df.loc[df["modem"] == "css", "ber"].max()),
        },
    }
    metrics_path = write_metrics(run, metrics)
    print(f"METRICS   : {metrics_path}")

    print()
    print("S13 — CSS vs MFSK A/B comparison (same payload, same RIR, same SNR)")
    print(
        f"seed {seed} | fs {FS_HZ:.0f} Hz | {PAYLOAD_BITS}-bit payload | rooms: {metrics['n_rooms']}"
    )
    piv = df.pivot_table(index="snr_db", columns="modem", values="ber")
    print(piv.round(6).to_string())
    print(
        f"\nAt SNR={WORKING_SNR_DB:.0f} dB: MFSK mean BER={mfsk_w:.4g}, "
        f"CSS mean BER={css_w:.4g} -> {'CSS' if winner == 'css' else 'MFSK'} is stronger."
    )


if __name__ == "__main__":
    main()
