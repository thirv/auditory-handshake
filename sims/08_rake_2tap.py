"""S15 -- RAKE vs no-RAKE A/B: MFSK BER vs SNR over the acoustic channel.

Standalone simulation (not imported by the test suite). It isolates the effect
of the RAKE front-end (:mod:`sim.rake`) on the legacy 4-FSK modem
(:class:`auditory_handshake.phy.MFSKModem`) under *identical* acoustic
conditions, to answer the open follow-up: *does collecting the multipath with a
RAKE receiver beat the reverb smearing tail?*

Fairness contract (the whole point of an A/B test):

- **Same payload**: one fixed random 512-bit message for every trial.
- **Same channel**: each selected room's deterministic RIR, synthesised once
  at the Sabine RT60.
- **Same noise**: RIR + additive noise are realised *once* per (room, SNR)
  point; that *identical* received waveform is demodulated by three receiver
  strategies, so the only thing that differs between them is the receiver:
    * ``no_rake`` -- plain MFSK modem, no channel mitigation (baseline);
    * ``rake``    -- 2-tap RAKE front-end, then the plain MFSK modem (the
                     modem's MMSE equaliser is OFF; RAKE *is* the mitigation);
    * ``mmse_ref`` -- the modem's built-in MMSE equaliser, shown for context.
  RAKE and MMSE are *competing* mitigations -- you apply one or the other, not
  both (RAKE stacked on MMSE double-processes the channel and collapses BER to
  ~0.5).

Per (room x SNR) the link is ``encode -> modulate -> RIR -> noise -> [RAKE]
-> demodulate`` and BER is measured on the decoded bits.

Outputs (PLAN ~12 contract, written to ``results/sim_08/<UTC-ts>/``):
config.json, tables/rake_ab_ber_vs_snr.csv, figs/rake_ab_per_room.png,
figs/rake_ab_summary.png, metrics.json, plus a stdout summary.

Run:  ``python sims/08_rake_2tap.py``   (or ``make sim-08``)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# Make the top-level ``sim`` package (repo root) importable when run directly
# (``python sims/08_rake_2tap.py``): the editable install only exposes
# ``auditory_handshake`` under ``src/``, not the sibling ``sim/`` module.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from auditory_handshake.acoustics.channel import compute_rir
from auditory_handshake.acoustics.room import calculate_rt60
from auditory_handshake.config import all_room_presets
from auditory_handshake.phy import MFSKConfig, MFSKModem
from auditory_handshake.results import create_run, write_metrics
from sim.rake import PathTap, find_paths, rake

# ---- simulation parameters ------------------------------------------------- #
FS_HZ = 48_000.0  # sample rate (Hz) -- project standard (= sim.rake.DEFAULT_FS)
PAYLOAD_BYTES = 64  # fixed message: 512 data bits
PAYLOAD_BITS = PAYLOAD_BYTES * 8

SNR_DBS = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
WORKING_SNR_DB = 10.0  # SNR at which we ask "does the link actually work?"

RAKE_MAX_PATHS = 2  # a 2-tap RAKE (direct path + strongest reflection)
RAKE_THRESHOLD_DB = -40.0  # keep RAKE taps down to this level below the peak

def _channel(
    waveform: np.ndarray, rir: np.ndarray, rng: np.random.Generator, snr_db: float
) -> np.ndarray:
    """Apply RIR + SNR-matched additive noise. Noise level derives from the
    post-RIR clean power, so the swept SNR is the receiver-side S/N ratio."""
    clean = np.convolve(waveform, rir, mode="full")[: waveform.size]
    ref_power = float(np.mean(np.abs(clean) ** 2))
    if ref_power <= 0.0:
        return clean
    sigma = math.sqrt(ref_power / (10.0 ** (snr_db / 10.0)))
    return clean + rng.normal(0.0, sigma, size=waveform.size)


def _demod_ber(
    mfsk: MFSKModem, tx_bits: np.ndarray, rx_waveform: np.ndarray, num_symbols: int
) -> float:
    """BER of the 4-FSK modem for one received waveform (mirrors sim_03)."""
    rx_symbols = mfsk.demodulate(rx_waveform).bits
    n = min(rx_symbols.size, num_symbols)
    rx_bits = MFSKModem.bits_from_symbols(np.asarray(rx_symbols[:n], dtype=np.int64))
    tx = np.asarray(tx_bits, dtype=np.int64)[: rx_bits.size]
    return float(np.mean(tx != rx_bits)) if rx_bits.size else 0.0


def _run_mfsk(
    payload: bytes,
    rir: np.ndarray,
    taps: list[PathTap],
    rng: np.random.Generator,
    snr_db: float,
) -> tuple[float, float]:
    """BER for one payload/RIR/SNR, with and without RAKE.

    The received waveform is realised *once*; both arms demodulate that exact
    same signal, so the only difference between the arms is the RAKE front-end.
    """
    bits_per_symbol = 2  # 4-FSK carries 2 data bits per symbol
    num_symbols = (PAYLOAD_BITS + bits_per_symbol - 1) // bits_per_symbol

    tx_symbols = MFSKModem.encode(payload, num_symbols)
    tx_bits = MFSKModem.bits_from_symbols(np.asarray(tx_symbols, dtype=np.int64))
    waveform = MFSKModem(MFSKConfig()).modulate(tx_bits).waveform

    # One shared received waveform; every strategy demodulates this exact signal.
    rx_waveform = _channel(waveform, rir, rng, snr_db)
    # no_rake: plain modem, no channel mitigation (the no-RAKE baseline).
    ber_no_rake = _demod_ber(MFSKModem(MFSKConfig()), tx_bits, rx_waveform, num_symbols)
    # rake: 2-tap RAKE front-end, then the plain modem (eq_rir OFF -- RAKE *is*
    # the channel mitigation here). RAKE and MMSE are competing strategies: you
    # apply one or the other, not both (RAKE + MMSE double-processes the channel
    # and collapses to ~0.5 BER).
    ber_rake = _demod_ber(MFSKModem(MFSKConfig()), tx_bits, rake(rx_waveform, taps), num_symbols)
    # mmse_ref: the modem's built-in MMSE equaliser, shown as a reference so the
    # RAKE win is measured against the project's existing channel equalizer too.
    ber_mmse = _demod_ber(MFSKModem(MFSKConfig(eq_rir=rir)), tx_bits, rx_waveform, num_symbols)
    return ber_no_rake, ber_rake, ber_mmse


def _pick_rooms() -> list[Any]:
    """Pick the two most informative rooms: quietest and most reverb-echoey,
    ranked by Sabine RT60 so we always get a 'small' and a 'large' room."""
    presets = all_room_presets()
    scored = sorted(presets, key=lambda p: calculate_rt60(p, formula="sabine", fs=FS_HZ))
    if len(scored) <= 2:
        return list(scored)
    return [scored[0], scored[-1]]


def run_ab(seed: int) -> pd.DataFrame:
    """Sweep every (room x SNR) point for both receivers; return a long table."""
    rows: list[dict[str, Any]] = []
    for room in _pick_rooms():
        rt60 = calculate_rt60(room, formula="sabine", fs=FS_HZ)
        rir = compute_rir(room, rt60, fs=FS_HZ)
        taps = find_paths(rir, threshold_db=RAKE_THRESHOLD_DB, max_paths=RAKE_MAX_PATHS, fs=FS_HZ)
        rng = np.random.default_rng(seed)
        payload = rng.bytes(PAYLOAD_BYTES)
        for snr_db in SNR_DBS:
            ber_no_rake, ber_rake, ber_mmse = _run_mfsk(
                payload, rir, taps, np.random.default_rng(seed), float(snr_db)
            )
            base = {
                "room": room.name,
                "snr_db": float(snr_db),
                "rt60_s": float(rt60),
                "n_taps": int(len(taps)),
            }
            rows.append({**base, "rx": "no_rake", "ber": ber_no_rake})
            rows.append({**base, "rx": "rake", "ber": ber_rake})
            rows.append({**base, "rx": "mmse_ref", "ber": ber_mmse})
    df = pd.DataFrame(rows)
    return df.sort_values(["room", "snr_db", "rx"]).reset_index(drop=True)

def _plot_per_room(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rooms = sorted(df["room"].unique())
    fig, axes = plt.subplots(1, len(rooms), figsize=(5.0 * len(rooms), 3.8), squeeze=False)
    for ax, room in zip(axes.ravel(), rooms, strict=False):
        sub = df[df["room"] == room]
        for rx, color, marker, ls in (
            ("no_rake", "tab:gray", "o", "-"),
            ("rake", "tab:green", "s", "--"),
            ("mmse_ref", "tab:blue", "^", ":"),
        ):
            s = sub[sub["rx"] == rx]
            ax.semilogy(s["snr_db"], s["ber"].clip(lower=1e-6), marker + ls, color=color, lw=2.0, label=rx)
        ax.axhline(1e-3, color="tab:gray", lw=1.0, ls=":", alpha=0.8)
        ax.set_title(f"{room}  ({int(sub['n_taps'].iloc[0])}-tap RAKE)", fontsize=10)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("S15 -- MFSK RAKE vs no-RAKE: BER vs SNR per room", y=1.02)
    fig.tight_layout()
    png = out_dir / "rake_ab_per_room.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG       : {png}")


def _plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rx, color, marker, ls in (
        ("no_rake", "tab:gray", "o", "-"),
        ("rake", "tab:green", "s", "--"),
        ("mmse_ref", "tab:blue", "^", ":"),
    ):
        sub = df[df["rx"] == rx].groupby("snr_db")["ber"].mean()
        ax.semilogy(sub.index, np.clip(sub.values, 1e-6, None), marker + ls, color=color, lw=2.0, label=rx)
    ax.axhline(1e-3, color="tab:gray", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER (mean over rooms)")
    ax.set_title(f"S15 -- MFSK RAKE vs no-RAKE: mean BER vs SNR  ({df['room'].nunique()} rooms, {PAYLOAD_BITS} bits/msg)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    png = out_dir / "rake_ab_summary.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"PNG       : {png}")


def main() -> None:
    seed = 20260912
    rooms = _pick_rooms()
    run = create_run(
        "sim_08",
        seed=seed,
        config={
            "fs_hz": FS_HZ,
            "payload_bytes": PAYLOAD_BYTES,
            "snr_dbs": [float(x) for x in SNR_DBS.tolist()],
            "rooms": [p.name for p in rooms],
            "modem": "mfsk",
            "receivers": ["no_rake", "rake", "mmse_ref (reference: built-in MMSE equalizer)"],
            "rake_max_paths": RAKE_MAX_PATHS,
            "rake_threshold_db": RAKE_THRESHOLD_DB,
        },
    )

    df = run_ab(seed)
    csv = run.tables_dir / "rake_ab_ber_vs_snr.csv"
    df.to_csv(csv, index=False)
    print(f"TABLE     : {csv}")

    _plot_per_room(df, run.figs_dir)
    _plot_summary(df, run.figs_dir)

    working = df[df["snr_db"] == WORKING_SNR_DB].groupby("rx")["ber"].mean()
    ber_no_rake = float(working.get("no_rake", float("nan")))
    ber_rake = float(working.get("rake", float("nan")))
    ber_mmse = float(working.get("mmse_ref", float("nan")))
    winner = "rake" if ber_rake <= ber_no_rake else "no_rake"
    improvement = (ber_no_rake / ber_rake) if (ber_no_rake > 0.0 and ber_rake > 0.0) else float("nan")

    metrics = {
        "n_rooms": int(df["room"].nunique()),
        "n_snr_points": int(df["snr_db"].nunique()),
        "payload_bits": int(PAYLOAD_BITS),
        "rake_max_paths": RAKE_MAX_PATHS,
        "working_snr_db": WORKING_SNR_DB,
        "mean_ber_at_working_snr": {"no_rake": ber_no_rake, "rake": ber_rake, "mmse_ref": ber_mmse},
        "lower_ber_at_working_snr": winner,
        "ber_ratio_no_rake_over_rake": improvement,
        "ber_range": {
            "no_rake_min": float(df.loc[df["rx"] == "no_rake", "ber"].min()),
            "no_rake_max": float(df.loc[df["rx"] == "no_rake", "ber"].max()),
            "rake_min": float(df.loc[df["rx"] == "rake", "ber"].min()),
            "rake_max": float(df.loc[df["rx"] == "rake", "ber"].max()),
            "mmse_ref_min": float(df.loc[df["rx"] == "mmse_ref", "ber"].min()),
            "mmse_ref_max": float(df.loc[df["rx"] == "mmse_ref", "ber"].max()),
        },
    }
    metrics_path = write_metrics(run, metrics)
    print(f"METRICS   : {metrics_path}")

    print()
    print("S15 -- MFSK RAKE vs no-RAKE A/B (same payload, same RIR, same noise)")
    print(
        f"seed {seed} | fs {FS_HZ:.0f} Hz | {PAYLOAD_BITS}-bit payload | "
        f"rooms: {metrics['n_rooms']} | RAKE <= {RAKE_MAX_PATHS} taps"
    )
    print(df.pivot_table(index="snr_db", columns="rx", values="ber").round(6).to_string())
    print(
        f"\nAt SNR={WORKING_SNR_DB:.0f} dB: no-RAKE BER={ber_no_rake:.4g}, "
        f"RAKE BER={ber_rake:.4g}, MMSE(ref)={ber_mmse:.4g} -> "
        f"{'RAKE is stronger than no-RAKE' if winner == 'rake' else 'no-RAKE is stronger'} "
        f"(no-RAKE/RAKE ratio={improvement:.3g}x). "
        f"{'RAKE collects the multipath and wins.' if winner == 'rake' else 'RAKE does not help here.'}"
    )


if __name__ == "__main__":
    main()
