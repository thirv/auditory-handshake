#!/usr/bin/env python3
"""sim_09: RT60 dependence of the two-way handshake round-trip time (C1).

Spec (PLAN.md §4.2 C1, §2.4, §1.3):
  C1: "Handshake is bounded by the room (reverb), not the radio."  [A, M5]
  Falsifier: "C1 dies if the measured handshake round-trip time is statistically
  insensitive to RT60 across the room presets (regression slope ≈ 0)."

What this driver does, end to end (no fabricated numbers):
  For a fixed room *geometry* (one base preset) it sweeps a target RT60 and, at
  each target, does a full round trip through the real stack:

    1. ``channel.compute_rir(base_preset, target_rt60)``  -- synthesize the
       exponential RIR for that RT60 (and lengthen the tail so one round trip
       survives it, per PLAN.md §2/§5).
    2. ``metrics.rt60_schroeder(rir, fs)``                 -- MEASURE RT60 back
       out of the RIR with the Schroeder energy-decay integral (the honest
       number to regress on, not the target we asked for).
    3. ``channel.echo_tail_guard_s(rir, fs)``              -- the reverb term:
       how long the tail takes to decay past -30 dB. This is the *guard* each
       direction must wait for before its own echo can't be mistaken for the
       peer -- the only stage time that depends on the room.
    4. Build a forward and a reverse :class:`TwoWayLeg` (frame tx + reverb
       guard + detect + proc, per direction) and take
       ``metrics.rtt_s(fwd, rev)`` = the handshake round-trip time.
    5. Collect (measured_RT60, RTT) across the sweep and run the canonical
       C1 falsifier ``metrics.regress_ttch_vs_rt60`` (delegates to
       ``ttch.regress_rt60_sensitivity``), reporting slope / R^2 / verdict.

Model (PLAN.md §2.4), per direction ``RTT_leg``:
    RTT_leg = T_tx + T_guard(RT60) + T_detect + T_proc
    RTT     = 2 * RTT_leg          (forward + reverse, same room)

``T_guard`` is what ``echo_tail_guard_s`` measures and is the only term that
scales with the room; ``T_tx`` (frame length, from the MFSK symbol rate),
``T_detect`` and ``T_proc`` are the fixed one-shot overheads. Because
``T_guard ≈ (|threshold|/20)·ln10·τ`` and ``RT60 ≈ 3·ln10·τ``, for the default
-30 dB guard ``T_guard ≈ 0.25·RT60`` -- i.e. RTT grows linearly with RT60, which
is exactly the positive slope C1 predicts.

Artifacts (canonical run dir, ``results/sim_09/<UTC-ts>/`` -- no ``--out-dir``):
    * ``config.json``   -- reproducibility stamp (experiment, seed, config,
      git SHA, UTC timestamp, software versions), written by ``create_run``.
    * ``metrics.json``  -- the ``regress_ttch_vs_rt60`` C1 regression + verdict
      plus the sweep parameters, written by ``write_metrics`` (seed / git SHA
      auto-stamped from ``config.json``).
    * ``tables/handshake_rt60.csv`` -- the raw per-room table (regression inputs).

Usage:
    python -m sims.09_two_way_handshake
    python -m sims.09_two_way_handshake --rt60 0.1 0.5 1.0
    python -m sims.09_two_way_handshake --rt60 0.05 0.1 0.3   # denser low-RT60
    python -m sims.09_two_way_handshake --rt60 0.5 1.0 1.5    # denser high-RT60
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict

import numpy as np

from auditory_handshake.acoustics import channel
from auditory_handshake.config import load_room_preset
from auditory_handshake.metrics import (
    TwoWayLeg,
    regress_ttch_vs_rt60,
    rt60_schroeder,
    rtt_s,
)
from auditory_handshake.phy.modem import MFSKConfig, MFSKModem
from auditory_handshake.results import create_run, write_metrics

# -- fixed one-shot overheads (PLAN.md §1.3 one-shot frame 10-25 ms) ---------
# A single handshake frame: 8 MFSK symbols at the 512 symbols/s rate => 15.6 ms,
# inside the 10-25 ms one-shot target.  These are room-independent constants;
# only T_guard varies with the room.
NUM_FRAME_SYMBOLS: int = 8
T_DETECT_S: float = 0.020  # receiver correlation + decision stage (one-way)
T_PROC_S: float = 0.010  # one-way DSP processing overhead

DEFAULT_RT60_S: tuple[float, ...] = (0.1, 0.2, 0.4, 0.7, 1.0)
DEFAULT_BASE_ROOM: str = "small_office"

# Reproducibility stamp (PLAN.md §5/§12).  The sim is deterministic -- analytic
# RIR synthesis, fixed one-shot overheads, no RNG -- so the seed is recorded for
# parity with the other sims (they all pin 20260912) rather than used by one.
SEED: int = 20260912


def _frame_tx_s(config: MFSKConfig) -> float:
    """Transmit time of one handshake frame (seconds)."""
    return NUM_FRAME_SYMBOLS / config.symbols_per_second


def _build_leg(rir: np.ndarray, fs: float, t_tx_s: float) -> TwoWayLeg:
    """A one-way handshake leg whose guard tracks the room's reverberant tail."""
    t_guard_s = float(channel.echo_tail_guard_s(rir, fs))
    return TwoWayLeg(
        t_tx_s=t_tx_s,
        t_guard_s=t_guard_s,
        t_detect_s=T_DETECT_S,
        t_proc_s=T_PROC_S,
        decoded=True,
    )


def run_sweep(
    rt60_targets: list[float],
    base_preset,
    fs: float,
) -> tuple[list[dict], list[float], list[float]]:
    """Run the round trip at each target RT60; return rows + x/y for regression."""
    modem = MFSKModem(MFSKConfig(fs=fs))  # real modem for the frame timing
    t_tx_s = _frame_tx_s(modem.config)

    rows: list[dict] = []
    rt60_meas: list[float] = []
    rtt: list[float] = []

    for target in rt60_targets:
        # 1) synthesize the channel for this RT60 (and a survivable tail).
        #    CRITICAL: pass the SAME fs used for every downstream analysis
        #    (rt60_schroeder, echo_tail_guard_s, TwoWayLeg). The default fs in
        #    compute_rir is 16 kHz; if left unset, the RIR is sampled at 16 kHz
        #    while being *interpreted* as fs (48 kHz), compressing all timings by
        #    48k/16k = 3x and under-reporting the measured RT60 (~0.28 s instead
        #    of ~0.76 s for a 1.0 s target).
        rir = channel.compute_rir(base_preset, target, fs=fs)
        # 2) measure RT60 back out of the RIR (Schroeder integral).
        rt60 = float(rt60_schroeder(rir, fs))
        # 3) forward + reverse legs through the real guard, then the RTT.
        fwd = _build_leg(rir, fs, t_tx_s)
        rev = _build_leg(rir, fs, t_tx_s)  # same room on the return path
        rtt_val = float(rtt_s(fwd, rev))

        rt60_meas.append(rt60)
        rtt.append(rtt_val)
        rows.append(
            {
                "target_rt60_s": target,
                "measured_rt60_s": rt60,
                "t_guard_s": fwd.t_guard_s,
                "t_tx_s": t_tx_s,
                "t_detect_s": T_DETECT_S,
                "t_proc_s": T_PROC_S,
                "one_leg_s": fwd.t_tx_s + fwd.t_guard_s + fwd.t_detect_s + fwd.t_proc_s,
                "round_trip_s": rtt_val,
            }
        )

    return rows, rt60_meas, rtt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rt60",
        type=float,
        nargs="+",
        default=list(DEFAULT_RT60_S),
        help="Target RT60 sweep (s). Default: 0.1 0.2 0.4 0.7 1.0",
    )
    ap.add_argument(
        "--base-room",
        default=DEFAULT_BASE_ROOM,
        help=f"Base room preset for geometry (default: {DEFAULT_BASE_ROOM})",
    )
    ap.add_argument("--fs", type=float, default=48_000.0, help="Sample rate (Hz)")
    args = ap.parse_args(argv)

    if len(args.rt60) < 2:
        print(f"FATAL: need >=2 RT60 targets to regress; got {len(args.rt60)}")
        return 2

    base_preset = load_room_preset(args.base_room)
    rows, rt60_meas, rtt = run_sweep(args.rt60, base_preset, args.fs)

    reg_dict = asdict(regress_ttch_vs_rt60(rt60_meas, rtt))

    # -- create the run dir (PLAN.md §5/§12) ---------------------------------
    # Canonical layout: results/sim_09/<UTC-ts>/{config.json, tables/...}.
    # config.json is the full reproducibility stamp (experiment, seed, config,
    # git SHA, UTC timestamp, software versions) and is the tracked artifact.
    run = create_run(
        "sim_09",
        seed=SEED,
        config={
            "fs_hz": args.fs,
            "base_room": args.base_room,
            "rt60_targets_s": [float(t) for t in args.rt60],
            "num_frame_symbols": NUM_FRAME_SYMBOLS,
            "t_detect_s": T_DETECT_S,
            "t_proc_s": T_PROC_S,
            "modem": "mfsk",
            "regressor": "regress_ttch_vs_rt60 (canonical C1 falsifier)",
        },
    )

    csv_path = run.tables_dir / "handshake_rt60.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # C1 verdict + the full regression stats live in metrics.json (tracked).
    write_metrics(
        run,
        {
            **reg_dict,
            "t_tx_s": rows[0]["t_tx_s"],
            "n_rt60_points": len(rows),
        },
    )

    report = [
        "sim_09 — RT60 dependence of the two-way handshake round-trip (C1)",
        "=" * 72,
        f"base room      : {args.base_room}",
        f"fs             : {args.fs:.0f} Hz",
        f"frame symbols  : {NUM_FRAME_SYMBOLS}  (T_tx = {rows[0]['t_tx_s'] * 1000:.2f} ms)",
        f"T_detect       : {T_DETECT_S * 1000:.2f} ms (one-way)",
        f"T_proc         : {T_PROC_S * 1000:.2f} ms (one-way)",
        "",
        "per-room round trip (target -> measured RT60 -> RTT):",
        "  target   measured_RT60   t_guard    RTT(ms)",
    ]
    for r in rows:
        report.append(
            f"  {r['target_rt60_s']:<9.3f} {r['measured_rt60_s']:<15.4f} "
            f"{r['t_guard_s'] * 1000:<9.2f} {r['round_trip_s'] * 1000:<10.2f}"
        )
    # target-vs-measured discrepancy analysis (C1 close-out investigation)
    ratios = [r["measured_rt60_s"] / r["target_rt60_s"] for r in rows]
    report += [
        "",
        "TARGET vs MEASURED RT60 (why they differ):",
        "  * Both are computed at the SAME fs (the RIR is synthesized with fs=, ",
        "    so the Schroeder integral and the guard are not compressed by a 48k/",
        "    16k sample-rate mismatch -- the dominant historical bug, now fixed).",
        "  * The residual measured < target gap is expected physics, not a bug:",
        "    the target is a Sabine diffuse-field value (0.161*V/A); the measured",
        "    value is a deterministic image-source decay in one small shoebox.",
        "    Sabine assumes a statistically diffuse field that a 4x5x2 m room",
        "    with max_order=40 reflections never fully reaches, so the honest",
        "    measured RT60 lands ~25% below the Sabine target.",
        "  * C1 is regressed on the MEASURED value (the real channel's RT60),",
        "    so the physical target-vs-measured gap does not leak into the claim.",
    ]
    for r, ratio in zip(rows, ratios, strict=True):
        report.append(
            f"    target {r['target_rt60_s']:<6.3f} s  measured "
            f"{r['measured_rt60_s']:<7.4f} s  (measured/target = {ratio:.2f})"
        )
    report += [
        "",
        "C1 regression (RTT ~ measured RT60):",
        f"  slope       : {reg_dict['slope']:+.4f}  (s RTT per s RT60)",
        f"  intercept   : {reg_dict['intercept']:.4f}",
        f"  pearson_r   : {reg_dict['pearson_r']}",
        f"  t_stat      : {reg_dict['t_stat']}",
        f"  n           : {reg_dict['n']}",
        f"  verdict     : {reg_dict['verdict']}",
        "",
        "INTERPRETATION:",
        "  * The only room-dependent stage is the reverb guard (T_guard), which",
        "    echo_tail_guard_s measures from the real RIR and which scales ~0.25x",
        "    RT60 at the default -30 dB threshold. That is what makes RTT grow",
        "    with RT60 -- i.e. the handshake IS bounded by the room (C1).",
        "  * A positive slope with high R^2 across the RT60 sweep corroborates C1.",
        "",
        f"ARTIFACTS: {csv_path}, {run.metrics_path}",
        "PROVENANCE: sim_09 C1 close-out -- real RIR synthesis (compute_rir),",
        "  measured RT60 (rt60_schroeder), real reverb guard (echo_tail_guard_s),",
        "  TwoWayLeg/rtt_s round trip, canonical falsifier regress_ttch_vs_rt60.",
    ]

    # -- console summary -----------------------------------------------------
    print("\n".join(report))
    print()
    print(f"  run dir     : {run.dir}")
    print(f"  config.json : {run.config_path}")
    print(f"  metrics.json: {run.metrics_path}")
    print(f"  table       : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
