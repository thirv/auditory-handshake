"""S12.2 — Motion & Doppler simulation (PLAN §2.3, §12).

Standalone simulation (not imported by the test suite). It exercises the
S12.1 motion utilities :mod:`auditory_handshake.acoustics.motion` —
:func:`doppler_factor`, :func:`moving_source`, and
:func:`measure_doppler_hz` — to demonstrate, on the §2.4 "drone arrives"
scenario, how a relative source velocity Doppler-shifts the received tone:

1. **Steady approach** — a pure tone is rendered from a source closing at a
   constant speed (the :func:`moving_source` S12.1 baseline). For each speed the
   received frequency shift is measured with :func:`measure_doppler_hz` and
   compared against the analytic prediction
   ``f_rec = f_src · c / (c - v)`` (:func:`doppler_factor`). The measured shift
   must track the prediction within tolerance — a direct end-to-end check of the
   S12.1 model against its own theory.
2. **Variable-speed approach** — a per-block closing-speed profile (a
   decelerating arrival, one closing speed per ``VAR_BLOCK_S`` block) is
   integrated cumulatively into a position/distance trajectory table, matching
   :func:`moving_source`'s piecewise-constant-per-block model.

Outputs (PLAN §12 contract, written to ``results/sim_02b/<UTC-ts>/``)
----------------------------------------------------------------------
- ``config.json``                        — inputs + environment (via ``create_run``).
- ``tables/doppler_vs_speed.csv``        — steady-state sweep (predicted vs. measured).
- ``tables/variable_speed_trajectory.csv`` — per-block trajectory + closing speed.
- ``figs/doppler_shift_curves.png``      — Doppler shift vs. approach speed.
- ``figs/variable_speed_trajectory.png`` — range closing + closing-speed profile.
- ``metrics.json``                       — summary metrics (via ``write_metrics``).
- A human-readable summary on stdout.

Run:  ``python sims/02b_motion_doppler.py``   (or ``make sim-02b``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auditory_handshake.acoustics.motion import (
    doppler_factor,
    measure_doppler_hz,
    moving_source,
)
from auditory_handshake.results import create_run, write_metrics

# ---- simulation parameters ------------------------------------------------- #
FS_HZ = 48_000.0  # sample rate (Hz) — project default (configs/defaults.yaml)
REF_TONE_HZ = 4_000.0  # steady-state reference tone (mid-CSS-band; 20 Hz << 4 kHz)

APPROACH_S = 0.25  # steady approach duration (s)
BLOCK_S = 0.05  # moving_source resampling block length (s)

STEADY_SPEEDS_MS = np.array([10.0, 20.0, 25.0, 30.0, 50.0, 80.0])  # closing speeds (m/s, + = in)
STEADY_TOL = 0.10  # measured-vs-predicted relative-error bound (fraction)
DRONE_BASELINE_MS = 25.0  # drone_arrival scenario speed (configs/scenarios/drone_arrival.yaml)

# §12.3 variable-speed "drone arrival" (§12 example: 10–30 m/s, ~500 m, 8 s):
# one closing speed per VAR_BLOCK_S block, decelerating from a 30 m/s cruise
# down to a 10 m/s final approach.
VAR_APPROACH_S = 8.0
VAR_BLOCK_S = 0.5
VAR_INITIAL_DIST_M = 500.0
N_VAR_BLOCKS = int(round(VAR_APPROACH_S / VAR_BLOCK_S))  # 16 blocks
VAR_SPEEDS_MS = np.linspace(30.0, 10.0, N_VAR_BLOCKS)  # per-block m/s


def _source_tone(freq_hz: float, duration_s: float, fs: float, margin: float = 1.5) -> np.ndarray:
    """A unit-amplitude 1-D tone, ``margin``x longer than needed for the render."""
    n = int(margin * duration_s * fs)
    t = np.arange(n, dtype=np.float64) / fs
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float64)


def _steady_state_sweep() -> pd.DataFrame:
    """Constant-speed approach: predicted vs. measured Doppler shift per speed."""
    rows: list[dict[str, Any]] = []
    sig = _source_tone(REF_TONE_HZ, APPROACH_S, FS_HZ)
    for v in STEADY_SPEEDS_MS:
        predicted_hz = REF_TONE_HZ * (doppler_factor(v) - 1.0)
        recv = moving_source(sig, FS_HZ, v, APPROACH_S, BLOCK_S)
        # measure_doppler_hz already returns the SHIFT (measured − nominal).
        measured_hz = float(measure_doppler_hz(recv, FS_HZ, REF_TONE_HZ))
        rel_err = abs(measured_hz - predicted_hz) / max(abs(predicted_hz), 1e-9)
        rows.append(
            {
                "speed_ms": float(v),
                "doppler_factor": float(doppler_factor(v)),
                "doppler_predicted_hz": float(predicted_hz),
                "freq_received_hz": float(REF_TONE_HZ + measured_hz),
                "doppler_measured_hz": measured_hz,
                "rel_err": float(rel_err),
                "check_ok": bool(rel_err <= STEADY_TOL),
            }
        )
    return pd.DataFrame(rows).set_index("speed_ms")


def _variable_speed_trajectory() -> pd.DataFrame:
    """Per-block §12.3 trajectory: the source closes ``speed_k · Δt`` each block.

    Mirrors :func:`moving_source`'s piecewise-constant-per-block model (one
    closing speed per ``VAR_BLOCK_S`` block), integrated cumulatively so the
    range closes monotonically from the initial distance — the physically
    correct per-block position table for a variable-speed arrival.
    """
    n = int(VAR_SPEEDS_MS.size)
    v = np.asarray(VAR_SPEEDS_MS, dtype=np.float64)
    close_m = v * VAR_BLOCK_S
    dist_start = np.empty(n, dtype=np.float64)
    dist_end = np.empty(n, dtype=np.float64)
    d = float(VAR_INITIAL_DIST_M)
    for k in range(n):
        dist_start[k] = d
        d -= float(close_m[k])
        dist_end[k] = d
    t_center = (np.arange(n, dtype=np.float64) + 0.5) * VAR_BLOCK_S
    return pd.DataFrame(
        {
            "block": np.arange(n),
            "t_center_s": t_center,
            "closing_speed_ms": v,
            "distance_start_m": dist_start,
            "distance_end_m": dist_end,
        }
    )


def main() -> None:
    """Run both motion sub-simulations and write the §12 contract."""
    from auditory_handshake.config import load_defaults

    try:
        seed = int(load_defaults().get("seed", 42))
    except Exception:  # pragma: no cover — defaults optional; seed 42 is the spec fallback
        seed = 42

    steady_df = _steady_state_sweep()
    var_df = _variable_speed_trajectory()

    run = create_run(
        "sim_02b",
        seed=seed,
        config={
            "fs_hz": FS_HZ,
            "ref_tone_hz": REF_TONE_HZ,
            "approach_s": APPROACH_S,
            "block_s": BLOCK_S,
            "steady_speeds_ms": [float(x) for x in STEADY_SPEEDS_MS.tolist()],
            "steady_tol": STEADY_TOL,
            "drone_baseline_ms": DRONE_BASELINE_MS,
            "var_speeds_ms": list(VAR_SPEEDS_MS.tolist()),
            "var_approach_s": VAR_APPROACH_S,
            "var_block_s": VAR_BLOCK_S,
            "var_initial_dist_m": VAR_INITIAL_DIST_M,
            "scenario": "drone_arrival",
        },
    )

    (run.tables_dir / "doppler_vs_speed.csv").write_text(steady_df.to_csv(float_format="%.6g"))
    (run.tables_dir / "variable_speed_trajectory.csv").write_text(
        var_df.to_csv(index=False, float_format="%.6g")
    )

    # ---- console summary ---------------------------------------------------- #
    print("S12.2 motion & Doppler — relative velocity -> Doppler shift (drone_arrival)")
    print("=" * 70)
    print(f"fs        : {FS_HZ:.0f} Hz          ref tone : {REF_TONE_HZ:.0f} Hz")
    n_steady_blocks = int(round(APPROACH_S / BLOCK_S))
    print(f"steady appr : {APPROACH_S} s ({n_steady_blocks} x {BLOCK_S} s blocks)  seed : {seed}")
    print("-" * 70)
    print(f"{'v (m/s)':>8} {'factor':>8} {'pred (Hz)':>12} {'meas (Hz)':>12} {'rel err':>9}  check")
    for v in STEADY_SPEEDS_MS:
        r = steady_df.loc[v]
        print(
            f"{v:8.1f} {r['doppler_factor']:8.5f} {r['doppler_predicted_hz']:12.1f} "
            f"{r['doppler_measured_hz']:12.1f} {r['rel_err']:8.3f}  "
            f"{'OK' if r['check_ok'] else 'FAIL'}"
        )
    n_ok = int(steady_df["check_ok"].sum())
    print("-" * 70)
    print(f"steady-state : {n_ok}/{len(steady_df)} speeds within {STEADY_TOL:.0%} of prediction")
    base = steady_df.loc[DRONE_BASELINE_MS]
    print(
        f"drone @ {DRONE_BASELINE_MS:.0f} m/s : {base['doppler_predicted_hz']:.1f} Hz shift "
        f"({base['doppler_factor']:.4f}x freq)"
    )
    print(
        f"var-speed    : {N_VAR_BLOCKS} blocks, {VAR_SPEEDS_MS[0]:.0f}->{VAR_SPEEDS_MS[-1]:.0f} m/s "
        f"from {VAR_INITIAL_DIST_M:.0f} m -> {var_df['distance_end_m'].iloc[-1]:.2f} m"
    )

    _plot_doppler_curves(steady_df, run.figs_dir)
    _plot_variable_trajectory(var_df, run.figs_dir)
    print(f"run dir   : {run.dir}")

    write_metrics(
        run,
        {
            "steady_speeds_n": int(len(steady_df)),
            "steady_speed_ms_min": float(steady_df.index.min()),
            "steady_speed_ms_max": float(steady_df.index.max()),
            "doppler_predicted_hz_max": float(steady_df["doppler_predicted_hz"].max()),
            "doppler_measured_hz_max": float(steady_df["doppler_measured_hz"].max()),
            "rel_err_max": float(steady_df["rel_err"].max()),
            "steady_check_n_ok": n_ok,
            "steady_all_ok": bool(n_ok == len(steady_df)),
            "var_initial_dist_m": float(VAR_INITIAL_DIST_M),
            "var_final_dist_m": float(var_df["distance_end_m"].iloc[-1]),
            "var_speeds_ms": [float(x) for x in VAR_SPEEDS_MS.tolist()],
        },
    )


def _plot_doppler_curves(steady_df: pd.DataFrame, out_dir: Path) -> None:
    """Doppler shift vs. approach speed: predicted curve, measured points, baseline."""
    import matplotlib

    matplotlib.use("Agg")  # must be set before pyplot is imported
    import matplotlib.pyplot as plt

    v_grid = np.linspace(0.0, float(max(STEADY_SPEEDS_MS)), 200)
    factor_grid = np.array([doppler_factor(float(v)) for v in v_grid])
    pred_grid = REF_TONE_HZ * (factor_grid - 1.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(v_grid, pred_grid, color="tab:gray", lw=1.5, ls="--", label="predicted (theory)")
    ax.plot(
        steady_df.index,
        steady_df["doppler_measured_hz"],
        "o-",
        color="tab:blue",
        lw=2.0,
        label="measured (moving_source)",
    )
    base = steady_df.loc[DRONE_BASELINE_MS]
    ax.axvline(DRONE_BASELINE_MS, color="tab:orange", lw=1.2, alpha=0.8)
    ax.scatter(
        [DRONE_BASELINE_MS],
        [base["doppler_predicted_hz"]],
        color="tab:orange",
        s=90,
        zorder=3,
        label=f"drone_arrival ({DRONE_BASELINE_MS:.0f} m/s)",
    )
    ax.set_xlabel("Approach speed v (m/s, + = closing)")
    ax.set_ylabel("Doppler shift (Hz)")
    ax.set_title(
        f"S12.2 — Doppler shift vs relative velocity  ({REF_TONE_HZ/1000:.0f} kHz tone, "
        f"vs 20 Hz subcarrier)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    png_path = out_dir / "doppler_shift_curves.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"PNG       : {png_path}")


def _plot_variable_trajectory(var_df: pd.DataFrame, out_dir: Path) -> None:
    """Variable-speed arrival: range closing per block + closing speed profile."""
    import matplotlib

    matplotlib.use("Agg")  # must be set before pyplot is imported
    import matplotlib.pyplot as plt

    t = np.concatenate([[0.0], (var_df["block"].to_numpy() + 1.0) * VAR_BLOCK_S])
    dist = np.concatenate([[float(VAR_INITIAL_DIST_M)], var_df["distance_end_m"].to_numpy()])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(t, dist, color="tab:blue", lw=2.0, label="distance (m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance from receiver (m)")
    ax.set_title("S12.2 — variable-speed arrival: range closing per block")
    ax2 = ax.twinx()
    ax2.plot(
        var_df["t_center_s"].to_numpy(),
        var_df["closing_speed_ms"].to_numpy(),
        "o-",
        color="tab:red",
        lw=1.4,
        ms=4,
        alpha=0.85,
        label="closing speed (m/s)",
    )
    ax2.set_ylabel("Closing speed (m/s)", color="tab:red")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    png_path = out_dir / "variable_speed_trajectory.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"PNG       : {png_path}")


if __name__ == "__main__":
    main()
