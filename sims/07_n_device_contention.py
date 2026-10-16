#!/usr/bin/env python3
"""M6 / C1 (PLAN.md §2.4, §1.4): N-device slotted-ALOHA contention inflates the
TTCH and **amplifies** the RT60 sensitivity — analytically, the ``k·RT60`` term
is multiplied by the contention turnaround factor ``T(N)``.

NOTE: this is a **purely analytical, deterministic model** (no stochastic
arrivals, no queueing, no BLER). Because TTCH is linear in RT60 for a fixed N,
the RT60-sensitivity regression is a perfect fit (r = 1.0) **by construction** —
a *theoretical bound* on RT60-sensitivity, not an empirical confirmation of
Claim C1 (the sim_06 verdict remains INCONCLUSIVE).

Builds on ``sims/06`` (single-link TTCH + RT60-sensitivity regression). Adds the
multi-device contention layer (``auditory_handshake.contention``) and, over the
PLAN.md §2.2 RT60 grid (``sims/02``), shows that:

* TTCH grows with the number of contending devices ``N`` via the clean-slot
  factor ``(1/(1-p_tx))**(N-1)`` (capped at ``n_slots_max``);
* the RT60-sensitivity regression stays positive and gets *steeper* as ``N``
  grows, because ``k·RT60`` sits inside each contended turnaround — the
  sensitivity factor ``d(TTCH)/d(RT60) = T·k`` grows with ``N``;
* at ``N = 1`` the model reduces exactly to the sim_06 baseline
  (``turnarounds = 2``) — a cross-simulation consistency check.

Purely analytical (no I/O); writes one ``ttch_contention_N{N}.csv`` per device
count (like sim_06) plus one PNG (TTCH-vs-RT60 per N, and RT60-sensitivity per
N) into ``results/sim_07_n_device_contention/`` (see ``default_out()``).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auditory_handshake.contention import (
    DEFAULT_N_SLOTS_MAX,
    DEFAULT_P_TX,
    expected_turnarounds,
    ttch_s,
)
from auditory_handshake.metrics.ttch import (
    DEFAULT_K,
    DEFAULT_T_PROC_S,
    regress_rt60_sensitivity,
)

# Canonical room presets, inlined here (there is no shared ``sims/_common``
# module in this repo). RT60 values are the sim_02 theoretical values
# (``rt60_theory_s``): cabin 0.102 s (sub-meter), corridor 0.247 s (small
# open), small_office 0.425 s, small_hall 0.711 s, large_hall 1.000 s.
# RT60_REPRO_S = 0.5 s is the C1 reproduction-room RT60.
ROOM_LABELS = ("cabin", "corridor", "small_office", "small_hall", "large_hall")
RT60_GRID_S = (0.102, 0.247, 0.425, 0.711, 1.000)
RT60_REPRO_S = 0.500


def default_out() -> Path:
    """Default output directory for this simulation's artifacts."""
    return Path("results") / "sim_07_n_device_contention"


# Contending-device counts to sweep (N = 1 is the uncontended baseline).
N_DEVICES: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 14, 20)


@dataclass
class ContentionResult:
    """Per-N contention TTCH result: N × RT60 table + RT60-sensitivity verdict.

    ``sensitivity_factor`` is ``d(TTCH)/d(RT60) = T·k`` in s/s (the clean,
    interpretable "ms of TTCH per ms of RT60"); ``slope_ms_per_s`` is the same
    quantity as returned by the existing ``regress_rt60_sensitivity`` metric
    (ms/s), plus its correlation strength and verdict (C1).
    """

    n_devices: int
    expected_turnarounds: float  # T = base·(1/(1-p_tx))**(N-1), capped
    sensitivity_factor: float  # T·k = dTTCH/dRT60 (s/s)
    rt60_s: list[float]
    room: list[str]
    ttch_ms: list[float]
    slope_ms_per_s: float | None
    pearson_r: float | None
    t_stat: float | None
    verdict: str
    ttch_ms_at_repro_room: float


def contention_ttch(
    n_devices: int,
    rt60_grid_s: Sequence[float] = RT60_GRID_S,
    *,
    p_tx: float = DEFAULT_P_TX,
    k: float = DEFAULT_K,
    t_proc_s: float = DEFAULT_T_PROC_S,
    n_slots_max: float = DEFAULT_N_SLOTS_MAX,
) -> ContentionResult:
    """Contention-inflated TTCH across the RT60 grid at a fixed device count N.

    TTCH per room comes from the existing TTCH model (via
    :func:`auditory_handshake.contention.ttch_s`) evaluated at the contention
    expected turnarounds; the C1 RT60-sensitivity verdict comes from the existing
    :func:`auditory_handshake.metrics.ttch.regress_rt60_sensitivity`.
    """
    rt60 = [float(x) for x in rt60_grid_s]
    turnarounds = expected_turnarounds(n_devices, p_tx=p_tx, n_slots_max=n_slots_max)
    ttch_ms = [
        ttch_s(n_devices, r, k=k, t_proc_s=t_proc_s, p_tx=p_tx, n_slots_max=n_slots_max) * 1000.0
        for r in rt60
    ]
    # C1 RT60-sensitivity: regress the per-room TTCH table (ms) against the
    # RT60 grid (s). TTCH is linear in RT60 for a fixed N, so this is a perfect
    # fit — slope (ms/s), correlation, t-statistic, and the verdict all come
    # from the shared metric rather than being hardcoded.
    sens = regress_rt60_sensitivity(rt60, ttch_ms)
    result = ContentionResult(
        n_devices=n_devices,
        expected_turnarounds=turnarounds,
        sensitivity_factor=turnarounds * k,
        rt60_s=rt60,
        room=list(ROOM_LABELS),
        ttch_ms=ttch_ms,
        slope_ms_per_s=sens.slope,
        pearson_r=sens.pearson_r,
        t_stat=sens.t_stat,
        verdict=sens.verdict,
        ttch_ms_at_repro_room=ttch_s(
            n_devices,
            RT60_REPRO_S,
            k=k,
            t_proc_s=t_proc_s,
            p_tx=p_tx,
            n_slots_max=n_slots_max,
        )
        * 1000.0,
    )
    return result


def _write_csv(path: Path, result: ContentionResult) -> None:
    """Write one N × RT60 table row (PLAN.md §2.4/§2.5) to ``path``."""
    rows = [
        {
            "rt60_s": rt60,
            "room": room,
            "ttch_ms": round(ttch, 3),
            "expected_turnarounds": round(result.expected_turnarounds, 4),
            "sensitivity_factor_s_per_s": round(result.sensitivity_factor, 4),
            "slope_ms_per_s": result.slope_ms_per_s,
            "pearson_r": result.pearson_r,
            "t_stat": result.t_stat,
            "verdict": result.verdict,
        }
        for rt60, room, ttch in zip(result.rt60_s, result.room, result.ttch_ms, strict=True)
    ]
    fieldnames = [
        "rt60_s",
        "room",
        "ttch_ms",
        "expected_turnarounds",
        "sensitivity_factor_s_per_s",
        "slope_ms_per_s",
        "pearson_r",
        "t_stat",
        "verdict",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(out: Path, results: list[ContentionResult]) -> None:
    """TTCH-vs-RT60 per N (left) and RT60-sensitivity per N (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        color = cmap(0.1 + 0.8 * i / max(1, len(results) - 1))
        ax1.plot(
            r.rt60_s,
            r.ttch_ms,
            marker="o",
            ms=3,
            lw=1.3,
            color=color,
            label=f"N={r.n_devices} (T≈{r.expected_turnarounds:.1f})",
        )
    ax1.set_xlabel("RT60 (s) — PLAN.md §2.2 room grid")
    ax1.set_ylabel("Contention TTCH (ms)")
    ax1.set_title("TTCH grows with RT60 for every N (analytical slotted-ALOHA model); steeper as N grows")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)

    positions = list(range(len(results)))
    colors = [cmap(0.1 + 0.8 * i / max(1, len(results) - 1)) for i in range(len(results))]
    ax2.bar(positions, [r.slope_ms_per_s for r in results], color=colors, alpha=0.88, width=0.6)
    for pos, r in zip(positions, results, strict=True):
        ax2.annotate(
            f"{r.verdict}\nT·k={r.sensitivity_factor:.2f}",
            (pos, r.slope_ms_per_s),
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax2.set_xticks(positions)
    ax2.set_xticklabels([str(r.n_devices) for r in results])
    ax2.set_xlabel("Number of contending devices N")
    ax2.set_ylabel("RT60-sensitivity  d(TTCH)/d(RT60)  (ms/s)")
    ax2.set_title("RT60-sensitivity d(TTCH)/d(RT60) grows with N (analytical model, not empirical)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ttch_contention_n_device.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: list[ContentionResult], out: Path) -> None:
    base = next((r for r in results if r.n_devices == 1), None)
    print(
        "N-dev |  T(exp)  | sens dTTCH/dRT60 (s/s) | r       | t    | verdict     | repro-room TTCH (ms)"
    )
    print("-" * 100)
    for r in results:
        r_txt = f"{r.pearson_r:.3f}" if r.pearson_r is not None else "  —  "
        if r.t_stat is None:
            t_txt = "     —  "
        elif r.t_stat > 1e6:  # exact (r = 1) linear fit → t-stat diverges; show as ∞
            t_txt = "∞ (r=1)"
        else:
            t_txt = f"{r.t_stat:8.3f}"
        print(
            f"{r.n_devices:<6} | {r.expected_turnarounds:7.3f} | "
            f"{r.sensitivity_factor:12.4f} | {r_txt:<7} | {t_txt:<8} | "
            f"{r.verdict:<13} | {r.ttch_ms_at_repro_room:18.3f}"
        )
    if base is not None:
        print("\nCross-check (C1): at N=1 the model reduces to the single-link " "baseline (T=2):")
        print(f"  N=1  TTCH @ repro room (0.5 s) = {base.ttch_ms_at_repro_room:.3f} ms")
    print(f"\nWrote {out}/ttch_contention_N*.csv and {out}/ttch_contention_n_device.png")


def main() -> None:
    out = default_out()
    out.mkdir(parents=True, exist_ok=True)
    results = [contention_ttch(n) for n in N_DEVICES]
    for r in results:
        _write_csv(out / f"ttch_contention_N{r.n_devices}.csv", r)
    _make_figure(out, results)
    _print_summary(results, out)


if __name__ == "__main__":
    main()
