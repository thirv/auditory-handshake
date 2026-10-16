"""S6 — Acoustic link budget: Shannon capacity vs. distance (PLAN §2.5, §10).

Standalone simulation (not imported by the test suite). It loads a scenario,
a TX device preset, and the project defaults from ``configs/``, derives the
free-space link inputs (noise floor, TX level, CSS-band bandwidth), and sweeps
the link budget across distance using
:mod:`auditory_handshake.metrics.link_budget`.

Outputs (PLAN §12 contract, written to ``results/sim_01/<UTC-ts>/``)
--------------------------------------------------------------------
- ``config.json``                     — inputs + environment (via ``create_run``).
- ``tables/link_budget.csv``          — the full sweep table.
- ``figs/capacity_vs_distance.png``   — capacity vs. distance.
- ``metrics.json``                    — summary metrics (via ``write_metrics``).
- A human-readable summary on stdout.

Run:  ``python sims/01_link_budget.py``   (or ``make sim-01``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auditory_handshake.config import load_defaults, load_device_preset, load_scenario
from auditory_handshake.metrics.link_budget import link_budget_sweep, noise_floor_db
from auditory_handshake.results import create_run, write_metrics

# Sweep grid (PLAN §2.5): 0.5 m (close) … 50 m (far), log-spaced.
DISTANCE_RANGE_M = (0.5, 50.0)
N_POINTS = 100


def derive_link_inputs() -> dict[str, Any]:
    """Derive the free-space link inputs from the configs.

    Noise floor = incoherent (power) sum of the scenario's noise sources in
    dB SPL; TX level = the device's max SPL; bandwidth = CSS band width (Hz).
    """
    scenario = load_scenario("quiet_office")
    device = load_device_preset("small_loudspeaker")
    defaults = load_defaults()

    band = list(defaults["audio_bands_hz"]["css"])
    return {
        "scenario": scenario.name,
        "device": device.name,
        "seed": int(scenario.seed),
        "tx_spl_db": float(device.max_spl_db),
        "noise_floor_db": float(noise_floor_db([ns.level_db_spl for ns in scenario.noise_sources])),
        "band_hz": band,
        "bandwidth_hz": float(band[1]) - float(band[0]),
    }


def main() -> None:
    """Run the link-budget sweep and write the §12 contract (run dir,
    ``config.json``, CSV table, figure, and ``metrics.json``)."""
    inp = derive_link_inputs()
    distances = np.logspace(np.log10(DISTANCE_RANGE_M[0]), np.log10(DISTANCE_RANGE_M[1]), N_POINTS)

    sw = link_budget_sweep(
        distances,
        tx_spl_db=inp["tx_spl_db"],
        noise_floor_db=inp["noise_floor_db"],
        bandwidth_hz=inp["bandwidth_hz"],
    )

    df = pd.DataFrame(
        {
            "distance_m": sw["distances_m"],
            "spl_rx_db": sw["spl_rx_db"],
            "snr_db": sw["snr_db"],
            "capacity_bps": sw["capacity_bps"],
        }
    )

    run = create_run(
        "sim_01",
        seed=inp["seed"],
        config={
            "scenario": inp["scenario"],
            "device": inp["device"],
            "tx_spl_db": inp["tx_spl_db"],
            "noise_floor_db": inp["noise_floor_db"],
            "band_hz": list(inp["band_hz"]),
            "bandwidth_hz": inp["bandwidth_hz"],
            "distance_range_m": list(DISTANCE_RANGE_M),
            "n_points": N_POINTS,
        },
    )

    csv_path = run.tables_dir / "link_budget.csv"
    df.to_csv(csv_path, index=False)

    # ---- console summary ---------------------------------------------------
    print("S6 acoustic link budget — capacity vs distance (free-space)")
    print("=" * 64)
    print(f"scenario  : {inp['scenario']}  (noise floor {inp['noise_floor_db']:.1f} dB SPL)")
    print(f"tx device : {inp['device']}  (max SPL {inp['tx_spl_db']:.1f} dB SPL @ 1 m)")
    band = inp["band_hz"]
    print(f"CSS band  : {band[0]/1000:.0f}-{band[1]/1000:.0f} kHz = {inp['bandwidth_hz']:.0f} Hz")
    print(
        f"range     : {DISTANCE_RANGE_M[0]:.1f} m ... {DISTANCE_RANGE_M[1]:.0f} m "
        f"({N_POINTS} log-spaced points)"
    )
    print("-" * 64)
    for probe in (1.0, 5.0, 10.0, 20.0, 50.0):
        row = df.loc[df["distance_m"].sub(probe).abs().idxmin()]
        print(
            f"  d = {row['distance_m']:6.2f} m -> {row['capacity_bps']/1000:8.1f} kbit/s "
            f"(SNR {row['snr_db']:5.1f} dB, RX {row['spl_rx_db']:5.1f} dB SPL)"
        )
    print("-" * 64)
    print(
        f"capacity  : {df['capacity_bps'].max()/1000:.1f} kbit/s (nearest) "
        f"... {df['capacity_bps'].min()/1000:.1f} kbit/s (farthest)"
    )
    print(f"CSV       : {csv_path}")

    _plot(distances, sw["capacity_bps"], run.figs_dir)
    print(f"run dir   : {run.dir}")

    write_metrics(
        run,
        {
            "n_points": int(len(distances)),
            "distance_m_min": float(distances[0]),
            "distance_m_max": float(distances[-1]),
            "capacity_bps_max": float(sw["capacity_bps"].max()),
            "capacity_bps_min": float(sw["capacity_bps"].min()),
            "snr_db_max": float(sw["snr_db"].max()),
            "snr_db_min": float(sw["snr_db"].min()),
        },
    )


def _plot(distances: np.ndarray, capacity_bps: np.ndarray, out_dir: Path) -> None:
    """Render the capacity-vs-distance curve (headless Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")  # must be set before pyplot is imported
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(distances, capacity_bps / 1000.0, color="tab:blue", lw=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Distance (m, log)")
    ax.set_ylabel("Channel capacity (kbit/s)")
    ax.set_title("S6 — Shannon capacity vs distance (free-space, quiet office)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / "capacity_vs_distance.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"PNG       : {png_path}")


if __name__ == "__main__":
    main()
