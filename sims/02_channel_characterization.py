"""S11 — Channel characterization: RT60 / delay spread / B50 per room preset.

Standalone simulation (not imported by the test suite). For every bundled
``configs/rooms/*.yaml`` preset it:

1. computes the theoretical RT60 with the Sabine equation
   (:func:`auditory_handshake.acoustics.calculate_rt60`);
2. synthesizes a deterministic RIR at that RT60 and measures the channel
   statistics the PHY design is matched against — Schroeder RT60, mean/RMS
   delay spread, B50/B90 — with the centralized
   :func:`auditory_handshake.acoustics.analyze_channel` binding (deterministic
   and hence reproducible without a seed);
3. checks the diffuse-field consistency relation ``tau_rms ≈ 0.205·RT60``.

Outputs (PLAN §12 contract, written to ``results/sim_02/<UTC-ts>/``)
--------------------------------------------------------------------
- ``config.json``                     — inputs + environment (via ``create_run``).
- ``tables/channel_characterization.csv`` — one row per room preset.
- ``figs/rt60_vs_tau_rms.png``        — RT60 vs τ_rms with the 0.205 line.
- ``metrics.json``                    — summary metrics (via ``write_metrics``).
- ``configs/phy_constraints.yaml``    — the published M2 constraint file
  (PLAN §7.2; also emitted as ``tables/phy_constraints.yaml`` in the run dir
  so the full contract stays self-contained).
- A human-readable summary on stdout.

Run:  ``python sims/02_channel_characterization.py``   (or ``make sim-02``)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from auditory_handshake.acoustics.channel import analyze_channel
from auditory_handshake.acoustics.room import calculate_rt60
from auditory_handshake.config import CONFIGS_DIR, all_room_presets
from auditory_handshake.results import create_run, write_metrics

#: Project standard sample rate for RIR generation (48 kHz; see
#: :mod:`auditory_handshake.acoustics.room` module docstring).
FS_HZ = 48000.0

#: Diffuse-field consistency bound for the τ_rms ≈ 0.205·RT60 check (±10%).
DIFFUSE_RATIO = 0.205
DIFFUSE_TOL = 0.10


def characterize_room(name: str) -> dict[str, Any]:
    """Characterize one room preset: theoretical RT60, deterministic RIR, metrics.

    Returns a flat dict with the preset name, the theoretical (Sabine) RT60,
    the measured Schroeder RT60, delay spreads, B50/B90, and the diffuse-field
    consistency flag.
    """
    preset = next(p for p in all_room_presets() if p.name == name)

    # 1. Theoretical (design) RT60 — Sabine, at the project's standard rate.
    rt60_theory = calculate_rt60(preset, formula="sabine", fs=FS_HZ)

    # 2. Deterministic RIR at that RT60 + measured channel statistics in one
    #    centralized, seed-free binding (reproducible by construction).
    channel = analyze_channel(preset, rt60_theory, fs=FS_HZ)
    rt60_measured = channel.rt60_schroeder
    tm = channel.tau_mean
    tr = channel.tau_rms
    b50_s = channel.b50
    b90_s = channel.b90

    # 4. Diffuse-field consistency: tau_rms should sit near 0.205·RT60.
    ratio = tr / rt60_measured if math.isfinite(rt60_measured) else float("inf")
    diffuse_ok = bool(
        math.isfinite(ratio) and abs(ratio - DIFFUSE_RATIO) / DIFFUSE_RATIO <= DIFFUSE_TOL
    )

    return {
        "room": name,
        "dims_m": [float(v) for v in preset.dimensions_m],
        "volume_m3": float(
            preset.dimensions_m[0] * preset.dimensions_m[1] * preset.dimensions_m[2]
        ),
        "rt60_theory_s": float(rt60_theory),
        "rt60_measured_s": float(rt60_measured),
        "tau_mean_s": float(tm),
        "tau_rms_s": float(tr),
        "b50_s": float(b50_s),
        "b90_s": float(b90_s),
        "tau_rms_over_rt60": float(ratio),
        "diffuse_check_ok": diffuse_ok,
    }


def main() -> None:
    """Sweep all bundled room presets and write the §12 contract + phy_constraints."""
    presets = all_room_presets()
    if not presets:
        raise SystemExit("no room presets found under configs/rooms/ — nothing to characterize")

    try:
        from auditory_handshake.config import load_defaults

        seed = int(load_defaults().get("seed", 42))
    except Exception:  # pragma: no cover — defaults optional; seed 42 is the spec fallback
        seed = 42

    rows = [characterize_room(p.name) for p in presets]

    # ---- table ------------------------------------------------------------- #
    df = pd.DataFrame(rows).set_index("room")
    df = df[
        [
            "dims_m",
            "volume_m3",
            "rt60_theory_s",
            "rt60_measured_s",
            "tau_mean_s",
            "tau_rms_s",
            "b50_s",
            "b90_s",
            "tau_rms_over_rt60",
            "diffuse_check_ok",
        ]
    ]

    run = create_run(
        "sim_02",
        seed=seed,
        config={
            "rooms": [p.name for p in presets],
            "fs_hz": FS_HZ,
            "rt60_formula": "sabine",
            "diffuse_ratio": DIFFUSE_RATIO,
            "diffuse_tol": DIFFUSE_TOL,
        },
    )

    csv_path = run.tables_dir / "channel_characterization.csv"
    df.to_csv(csv_path, float_format="%.6g")

    # ---- phy_constraints.yaml (M2 deliverable, PLAN §7.2) ------------------- #
    constraints = {
        "version": 1,
        "generated_by": "sims/02_channel_characterization.py",
        "fs_hz": FS_HZ,
        "seed": seed,
        "rooms": {
            row["room"]: {
                "rt60_s": round(row["rt60_theory_s"], 6),
                "tau_mean_s": round(row["tau_mean_s"], 9),
                "tau_rms_s": round(row["tau_rms_s"], 9),
                "b50_s": round(row["b50_s"], 9),
                "b90_s": round(row["b90_s"], 9),
            }
            for row in rows
        },
    }
    constraints_yaml = yaml.safe_dump(constraints, sort_keys=False)
    (run.tables_dir / "phy_constraints.yaml").write_text(constraints_yaml)

    # Publish the constraint file in-tree (the M2 placeholder is documented as
    # "regenerated via make sim-02" — overwriting it is the intended flow).
    phy_path = CONFIGS_DIR / "phy_constraints.yaml"
    phy_path.write_text(constraints_yaml)

    # ---- console summary ----------------------------------------------------- #
    print("S11 — channel characterization (RT60 / delay spread / B50 per room)")
    print("=" * 78)
    print(f"seed {seed} | fs {FS_HZ:.0f} Hz | RT60 formula: Sabine | rooms: {len(presets)}")
    print("-" * 78)
    print(
        f"{'room':<16} {'V(m3)':>7} {'RT60_t':>8} {'RT60_m':>8} "
        f"{'tau_mean':>8} {'tau_rms':>8} {'B50':>8} {'B90':>8}  {'tau/RT60':>8}  {'diff.':>5}"
    )
    for row in rows:
        print(
            f"{row['room']:<16} {row['volume_m3']:>7.1f} "
            f"{row['rt60_theory_s']*1000:>7.1f} {row['rt60_measured_s']*1000:>7.1f} "
            f"{row['tau_mean_s']*1000:>7.2f} {row['tau_rms_s']*1000:>7.2f} "
            f"{row['b50_s']*1000:>7.2f} {row['b90_s']*1000:>7.2f} "
            f"{row['tau_rms_over_rt60']:>8.4f}  {'ok' if row['diffuse_check_ok'] else 'WARN':>5}"
        )
    print("(times in ms)")
    print("-" * 78)
    print(f"CSV       : {csv_path}")
    print(f"constraints: {phy_path}")

    _plot(df, run.figs_dir)
    print(f"run dir   : {run.dir}")

    write_metrics(
        run,
        {
            "n_rooms": len(rows),
            "rt60_s_min": float(min(r["rt60_theory_s"] for r in rows)),
            "rt60_s_max": float(max(r["rt60_theory_s"] for r in rows)),
            "tau_rms_s_min": float(min(r["tau_rms_s"] for r in rows)),
            "tau_rms_s_max": float(max(r["tau_rms_s"] for r in rows)),
            "b50_s_min": float(min(r["b50_s"] for r in rows)),
            "b50_s_max": float(max(r["b50_s"] for r in rows)),
            "diffuse_check_pass": sum(1 for r in rows if r["diffuse_check_ok"]),
            "diffuse_check_fail": sum(1 for r in rows if not r["diffuse_check_ok"]),
        },
    )


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    """Render RT60 vs τ_rms per room with the diffuse-field 0.205 guide."""
    import matplotlib

    matplotlib.use("Agg")  # must be set before pyplot is imported
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    rt60 = (df["rt60_theory_s"] * 1000).to_numpy()
    tau = (df["tau_rms_s"] * 1000).to_numpy()
    ax.scatter(rt60, tau, s=60, color="tab:blue", zorder=3)
    for name in df.index:
        loc = df.index.get_loc(name)
        ax.annotate(
            name,
            (rt60[loc], tau[loc]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.plot(
        [0.0, rt60.max()],
        [0.0, DIFFUSE_RATIO * rt60.max()],
        "--",
        color="gray",
        lw=1,
        label=f"diffuse field tau_rms = {DIFFUSE_RATIO}*RT60",
    )
    ax.set_xlabel("Theoretical RT60 (ms)")
    ax.set_ylabel("tau_rms (ms)")
    ax.set_title("S11 — channel characterization (RT60 vs delay spread)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    png_path = out_dir / "rt60_vs_tau_rms.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"PNG       : {png_path}")


if __name__ == "__main__":
    main()
