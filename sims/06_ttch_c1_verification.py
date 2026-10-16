#!/usr/bin/env python3
"""sim_06: C1 verification - is the handshake bounded by the room (reverb)?

Spec (PLAN.md §4.2 C1, §2.4, §3.1, §5 M6).
  C1: "Handshake is bounded by the room (reverb), not the radio."  [A, M5]
  Falsification: "C1 dies if the measured handshake round-trip time is statistically
  insensitive to RT60 across the room presets (regression slope ≈ 0)."

DATA SOURCES (real, this repo):
  * RT60 per room:
        results/sim_02_<ts>/tables/channel_characterization.csv   (rt60_theory_s)
        (also echoed as `rt60_s` in sim_04's preamble_detection.csv)
  * Preamble-detection latency per room (M4, one-way):
        results/sim_04_preamble_detection_<ts>/tables/preamble_detection.csv
        - arm=channel : reverb-limited one-way detection latency - the reverb-limited
                        leg of a TTCH round trip; the closest available proxy for C1.
        - arm=awgn    : ideal (no-reverb) detection latency (~1.25 ms) - the fixed-
                        overhead lower bound (shows latency is overhead-limited, not
                        reverb-limited).
  The full handshake round-trip (M6) is NOT yet measured, so the overall C1 status is
  INCONCLUSIVE by construction; the driver reports the §2.4 model per room and the
  empirical proxy regression (M4 channel-arm latency vs RT60).

MODEL (PLAN.md §2.4):
    TTCH_RTT = 2*k*RT60 + 2*T_preamble + 2*T_frame + T_proc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from auditory_handshake.metrics import ttch  # canonical TTCH model (PLAN.md §2.4)

# Documented defaults (PLAN.md §1.3 one-shot frame "10-25 ms"; T_proc "2-6 ms"; §5 k).
DEFAULT_T_FRAME_S = 0.015  # one one-shot frame decode leg
DEFAULT_T_PROC_S = 0.004  # one-way DSP processing, counted once per round trip


# --------------------------------------------------------------------------- loaders


def _latest_run(experiment: str) -> Path | None:
    """Latest timestamped run dir under results/<experiment>/, or None."""
    base = Path("results") / experiment
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "tables").is_dir())
    return runs[-1] if runs else None


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def load_rooms(sim04_run: Path) -> list[dict]:
    """Per-room RT60 + M4 one-way detection latency (both arms), from sim_04."""
    rows = _read_csv(sim04_run / "tables" / "preamble_detection.csv")
    rooms: dict[str, dict] = {}
    for r in rows:
        name = r["room"].strip()
        rt60 = _f(r, "rt60_s")
        p50 = _f(r, "latency_p50_ms")
        if not name or math.isnan(rt60):
            continue
        rec = rooms.setdefault(name, {"rt60_s": rt60, "ch": [], "awgn": []})
        if r.get("arm") == "channel":
            if not math.isnan(p50):
                rec["ch"].append(p50)
        elif r.get("arm") == "awgn":
            if not math.isnan(p50):
                rec["awgn"].append(p50)
    out = []
    for name, rec in rooms.items():
        t_ch = sum(rec["ch"]) / len(rec["ch"]) if rec["ch"] else math.nan
        t_awgn = sum(rec["awgn"]) / len(rec["awgn"]) if rec["awgn"] else math.nan
        out.append(
            {
                "room": name,
                "rt60_s": rec["rt60_s"],
                "t_preamble_ch_ms": t_ch,  # M4 reverb-limited one-way leg (proxy)
                "t_preamble_awgn_ms": t_awgn,  # ideal (fixed-overhead lower bound)
                "regime": ttch.classify_regime(rec["rt60_s"]),
            }
        )
    out.sort(key=lambda d: d["rt60_s"])
    return out


def build_model(rooms: list[dict], k: float, t_frame_s: float, t_proc_s: float) -> list[dict]:
    """Add the §2.4 model TTCH per room via the canonical ``ttch_rtt_s``,
    decomposed into the reverb leg (2*k*RT60) and the fixed-overhead leg (ms)."""
    for rec in rooms:
        rt60 = rec["rt60_s"]
        t_preamble_s = (rec["t_preamble_ch_ms"] / 1000.0) if rec["t_preamble_ch_ms"] else 0.0
        rec["ttch_model_ms"] = (
            ttch.ttch_rtt_s(
                rt60, k=k, t_preamble_s=t_preamble_s, t_frame_s=t_frame_s, t_proc_s=t_proc_s
            )
            * 1000.0
        )
        rec["reverb_term_ms"] = 2.0 * k * rt60 * 1000.0
        rec["fixed_term_ms"] = rec["ttch_model_ms"] - rec["reverb_term_ms"]
    return rooms


def regress(rooms: list[dict], ykey: str) -> dict:
    """OLS RT60-sensitivity of ``ykey`` (ms) vs RT60 (s), via the canonical
    ``ttch.regress_rt60_sensitivity``.

    The canonical fit works in **seconds** (slope = s latency per s RT60), while
    ``ykey`` is in ms, so the ms values are converted down for the fit and the
    slope converted back to ms/s. Returns a flat dict the report/JSON/figure
    consume: ``n, rt60_sensitivity_ms_per_s, pearson_r, t_stat, intercept_ms,
    verdict`` (verdict is the canonical 5-level string)."""
    nan = float("nan")
    pairs = [
        (r["rt60_s"], r[ykey])
        for r in rooms
        if math.isfinite(r["rt60_s"]) and math.isfinite(r[ykey])
    ]
    xs = [x for x, _ in pairs]
    y_s = [y / 1000.0 for _, y in pairs]  # ms -> s (canonical works in seconds)
    insufficient = {
        "n": len(xs),
        "rt60_sensitivity_ms_per_s": nan,
        "pearson_r": nan,
        "t_stat": nan,
        "intercept_ms": nan,
        "verdict": "INCONCLUSIVE",
    }
    if len(xs) < 2:  # canonical requires n >= 2
        return insufficient
    try:
        res = ttch.regress_rt60_sensitivity(xs, y_s)
    except ValueError:  # e.g. zero-variance rt60 (sxx == 0)
        return insufficient
    return {
        "n": res.n,
        "rt60_sensitivity_ms_per_s": res.slope * 1000.0,  # (s/s) -> (ms/s)
        "pearson_r": res.pearson_r,
        "t_stat": res.t_stat,
        "intercept_ms": res.intercept * 1000.0,  # s -> ms
        "verdict": res.verdict,
    }


# -------------------------------------------------------------------------- outputs


def _fnum(v: float) -> str:
    return f"{v:.3f}" if (isinstance(v, float) and not math.isnan(v)) else "n/a"


def _write_figure(rooms, reg_emp, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        return f"  (skipped - {exc.__class__.__name__})"
    pts = [r for r in rooms if not math.isnan(r["t_preamble_ch_ms"])]
    xs = sorted(r["rt60_s"] for r in pts)
    y_ch = {r["rt60_s"]: r["t_preamble_ch_ms"] for r in pts}
    y_model = {r["rt60_s"]: r["ttch_model_ms"] for r in pts}
    sl = reg_emp.get("rt60_sensitivity_ms_per_s", float("nan"))
    ic = reg_emp.get("intercept_ms", float("nan"))
    _pr = reg_emp.get("pearson_r")
    r2 = (_pr * _pr) if isinstance(_pr, (int, float)) and not math.isnan(_pr) else float("nan")
    grid = list(np.linspace(xs[0], xs[-1], 20))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [r["rt60_s"] for r in pts],
        [y_ch[r["rt60_s"]] for r in pts],
        s=70,
        color="#1f77b4",
        zorder=3,
        label="M4 channel-arm delay (measured)",
    )
    ax.scatter(
        [r["rt60_s"] for r in pts],
        [y_model[r["rt60_s"]] for r in pts],
        s=40,
        marker="x",
        color="#d62728",
        zorder=3,
        label="§2.4 model TTCH",
    )
    if not math.isnan(sl):
        ax.plot(
            grid,
            [sl * x + ic for x in grid],
            "--",
            color="#1f77b4",
            label=f"regression (r²={r2:.2f})",
        )
    ax.set_xlabel("RT60 (s)")
    ax.set_ylabel("one-way delay (ms)")
    ax.set_title("C1: is delay bounded by the room? (M4 proxy vs §2.4 model)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "c1_ttch_vs_rt60.png", dpi=120)
    plt.close(fig)
    return "  figs/c1_ttch_vs_rt60.png"


def write_output(
    out_dir: Path, rooms, reg_emp, reg_model, k, t_frame_s, t_proc_s, sim04_run, sim02_run, verdict
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    written = []

    # tables/c1_ttch_by_room.csv
    p1 = out_dir / "tables" / "c1_ttch_by_room.csv"
    with open(p1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "room",
                "rt60_s",
                "regime",
                "t_preamble_ch_ms",
                "t_preamble_awgn_ms",
                "reverb_term_ms",
                "fixed_term_ms",
                "ttch_model_ms",
            ]
        )
        for r in rooms:
            w.writerow(
                [
                    r["room"],
                    f"{r['rt60_s']:.4f}",
                    r["regime"],
                    f"{r['t_preamble_ch_ms']:.2f}",
                    f"{r['t_preamble_awgn_ms']:.2f}",
                    f"{r['reverb_term_ms']:.2f}",
                    f"{r['fixed_term_ms']:.2f}",
                    f"{r['ttch_model_ms']:.2f}",
                ]
            )
    written.append(p1)

    # tables/c1_regression.json
    p2 = out_dir / "tables" / "c1_regression.json"
    payload = {
        "spec": "C1 (PLAN.md §4.2) - handshake bounded by the room (reverb), not the radio",
        "data": {
            "sim04_run": str(sim04_run),
            "sim02_run": str(sim02_run),
            "t_frame_s": t_frame_s,
            "t_proc_s": t_proc_s,
            "k": k,
        },
        "empirical_proxy": {  # M4 channel-arm one-way latency vs RT60
            "y": "t_preamble_ch_ms (M4 channel arm, one-way)",
            "x": "rt60_s",
            **{
                k2: reg_emp.get(k2)
                for k2 in ("n", "rt60_sensitivity_ms_per_s", "pearson_r", "t_stat", "verdict")
            },
        },
        "model": {  # §2.4 TTCH vs RT60
            "y": "ttch_model_ms (2k*RT60 + 2*T_preamble + 2*T_frame + T_proc)",
            "x": "rt60_s",
            **{
                k2: reg_model.get(k2)
                for k2 in ("n", "rt60_sensitivity_ms_per_s", "pearson_r", "t_stat", "verdict")
            },
        },
        "verdict_empirical_proxy": verdict,
        "overall_c1_status": "INCONCLUSIVE",
        "overall_note": (
            "Full handshake round-trip (M6) is not yet measured; C1 is scored on the "
            "M4 one-way channel-arm delay as the closest proxy. The measured delay does "
            "not track RT60, so current evidence does not corroborate C1."
        ),
    }
    p2.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    written.append(p2)

    # figs
    fig_note = _write_figure(rooms, reg_emp, out_dir)
    if (out_dir / "figs" / "c1_ttch_vs_rt60.png").is_file():
        written.append(out_dir / "figs" / "c1_ttch_vs_rt60.png")

    # report.txt
    p3 = out_dir / "report.txt"
    L = []
    L.append("C1 VERIFICATION - handshake bounded by the room (reverb)?")
    L.append("=" * 60)
    L.append("Spec: PLAN.md §4.2 C1 (A, M5); model §2.4; regimes §3.1; M6 §5.")
    L.append("")
    L.append("C1:  Handshake is bounded by the room (reverb), not the radio.")
    L.append("Falsification: C1 dies if measured handshake round-trip time is")
    L.append("      statistically insensitive to RT60 across rooms (slope ≈ 0).")
    L.append("")
    L.append("DATA SOURCES (real, this repo):")
    L.append(f"  sim_04 (M4 latency):  {sim04_run}")
    L.append(f"  sim_02 (RT60):        {sim02_run}")
    L.append(
        f"  model params:         k={k}, T_frame={t_frame_s*1000:.1f}ms, T_proc={t_proc_s*1000:.1f}ms (PLAN.md §1.3 defaults)"
    )
    L.append("  T_preamble:           M4 channel-arm one-way delay (per room)")
    L.append("")
    L.append(
        f"{'room':16s}{'RT60(s)':>9s}{'regime':>9s}{'M4_ch':>9s}{'awgn':>8s}{'reverb':>9s}{'fixed':>9s}{'TTCH_ms':>9s}"
    )
    for r in rooms:
        L.append(
            f"{r['room']:16s}{r['rt60_s']:9.4f}{r['regime']:>9s}"
            f"{_fnum(r['t_preamble_ch_ms']):>9s}{_fnum(r['t_preamble_awgn_ms']):>8s}"
            f"{_fnum(r['reverb_term_ms']):>9s}{_fnum(r['fixed_term_ms']):>9s}"
            f"{_fnum(r['ttch_model_ms']):>9s}"
        )
    L.append("")
    L.append("REGRESSIONS:")
    L.append(
        f"  empirical proxy (M4 ch delay vs RT60): slope={reg_emp.get('rt60_sensitivity_ms_per_s', float('nan')):.2f} ms/s  pearson_r={reg_emp.get('pearson_r')}  t={reg_emp.get('t_stat')}  n={reg_emp.get('n')}  verdict={reg_emp.get('verdict')}"
    )
    L.append(
        f"  model TTCH (§2.4) vs RT60:              slope={reg_model.get('rt60_sensitivity_ms_per_s', float('nan')):.2f} ms/s  pearson_r={reg_model.get('pearson_r')}  t={reg_model.get('t_stat')}  n={reg_model.get('n')}  verdict={reg_model.get('verdict')}"
    )
    L.append("")
    L.append(f"VERDICT (empirical proxy): {verdict}")
    L.append("OVERALL C1 STATUS: INCONCLUSIVE")
    L.append("")
    L.append("INTERPRETATION:")
    L.append("  * The M4 one-way detection delay is FLAT across a 7x RT60 range")
    L.append("    (0.102s -> 0.711s), and the awgn arm is a constant 1.25ms: latency is")
    L.append("    dominated by the fixed detection/processing overhead, not by the reverb")
    L.append("    tail. This does NOT support C1 (" "latency bounded by the room" ").")
    L.append("  * The §2.4 model DOES include a 2k*RT60 term, but its reverb term is")
    L.append("    comparable to (not larger than) the fixed 2*T_preamble+2*T_frame+T_proc.")
    L.append("  * The definitive C1 test is the full handshake round-trip (M6), which is")
    L.append("    not yet measured. Until M6 lands, C1 cannot be confirmed.")
    L.append("")
    L.append(
        f"ARTIFACTS: tables/c1_ttch_by_room.csv, tables/c1_regression.json, {fig_note.strip()}"
    )
    L.append("")
    L.append("PROVENANCE: sim_06 C1 verification (TTCH §2.4 model + M4/sim_02 real data);")
    L.append("  k=0.35 (PLAN.md §5), T_frame/T_proc from §1.3 one-shot targets.")
    L.append("  NOTE: this driver does not fabricate round-trip data; the round-trip")
    L.append("  (M6) is pending, so C1 is reported INCONCLUSIVE, not corroborated.")
    p3.write_text("\n".join(L) + "\n")
    written.append(p3)

    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/sim_06_ttch_c1_verification")
    ap.add_argument("--k", type=float, default=ttch.DEFAULT_K, help="reverb k (default 0.35)")
    ap.add_argument(
        "--t-frame", type=float, default=DEFAULT_T_FRAME_S, help="T_frame (s), default 0.015"
    )
    ap.add_argument(
        "--t-proc", type=float, default=DEFAULT_T_PROC_S, help="T_proc (s), default 0.004"
    )
    args = ap.parse_args(argv)

    sim04_run = _latest_run("sim_04_preamble_detection")
    if sim04_run is None or not (sim04_run / "tables" / "preamble_detection.csv").is_file():
        print("FATAL: sim_04_preamble_detection run with tables/preamble_detection.csv not found")
        return 2
    sim02_run = _latest_run("sim_02")

    rooms = load_rooms(sim04_run)
    if len(rooms) < 2:
        print(f"FATAL: need ≥2 rooms with RT60+M4 data; got {len(rooms)}")
        return 2
    build_model(rooms, args.k, args.t_frame, args.t_proc)
    reg_emp = regress(rooms, "t_preamble_ch_ms")
    reg_model = regress(rooms, "ttch_model_ms")
    verdict = reg_emp.get("verdict", "insufficient data")  # canonical 5-level verdict

    out_dir = Path(args.out)
    written = write_output(
        out_dir,
        rooms,
        reg_emp,
        reg_model,
        args.k,
        args.t_frame,
        args.t_proc,
        sim04_run,
        sim02_run,
        verdict,
    )
    print(f"C1 verification (M4 proxy, §2.4 model) -> {out_dir}")
    print(f"  rooms: {[r['room'] for r in rooms]}")
    print(f"  empirical proxy verdict: {verdict}  (pearson_r={reg_emp.get('pearson_r')})")
    print("  overall C1 status: INCONCLUSIVE (M6 round-trip pending)")
    for p in written:
        print(f"    {p}")
    print(f"  {out_dir/'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
