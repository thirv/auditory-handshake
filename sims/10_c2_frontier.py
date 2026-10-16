"""S17.2 — C2 frontier: goodput vs. audibility cost (sim 10).

For every candidate band in the ``c2_frontier.yaml`` config, build a
4-tone MFSK design point that spans the band, **measure** its real BER/BLER
under the channel, and keep the *evidence-corrected* design point (goodput
computed with the measured BER).  The C2 frontier is then extracted with the
verified ``metrics.rate`` API:

* :func:`nondominated` — the Pareto front (maximise goodput, minimise the
  aggregate audibility cost).
* :func:`trace_optimum_over_lambda` — sweep λ over the config's grid and
  record the optimal design at each λ.
* :func:`c2_verdict` — the C2 non-degeneracy criterion (≥ 2 distinct frontier
  points ⇒ ``CORROBORATED``).

Everything runs on the real MFSK modem (no hand-rolled numerics), is
deterministic (fixed seed), and prints a readable report.

Usage
-----
    python sims/10_c2_frontier.py

Channel note
------------
``FrontierConfig`` carries a single ``snr_db`` (the SNR axis anchor) and a
named room, but defining the frontier does **not** require a room impulse
response: each design point is measured over an AWGN channel at ``snr_db``
(``rir=None``), a first-class channel mode of :func:`measure_design_bler`.
"""

from __future__ import annotations

import sys
from dataclasses import replace

from auditory_handshake.config import FrontierConfig, load_frontier
from auditory_handshake.metrics.rate import (
    DesignPoint,
    c2_verdict,
    make_perceptual_cost_fn,
    nondominated,
    trace_optimum_over_lambda,
)
from auditory_handshake.phy.design import (
    DesignBlerResult,
    DesignInfeasibleError,
    measure_design_bler,
)

NUM_SYMBOLS = 256          # symbols transmitted per measurement (default)
SEED = 0                  # fixed seed → deterministic, reproducible frontier


def _build_design_point(cfg: FrontierConfig, band_lo: float, band_hi: float) -> DesignPoint:
    """Construct a 4-tone MFSK design point spanning ``[band_lo, band_hi]``.

    Four tones are placed evenly across the band so the point is
    band-feasible for the MFSK mapping (``base_hz = band_lo``,
    ``top_hz = band_hi``).  ``ber`` is a placeholder until it is replaced by
    the measured value.
    """
    span = band_hi - band_lo
    if span <= 0.0:
        raise DesignInfeasibleError(f"band [{band_lo:g}, {band_hi:g}] must have positive width")
    delta = span / 3.0
    occupied = (band_lo, band_lo + delta, band_lo + 2.0 * delta, band_hi)
    return DesignPoint(
        n_bands=4,
        bandwidth_per_band_hz=delta,
        snr_db=cfg.snr_db,
        level_db=cfg.spl_db,
        occupied_frequencies_hz=occupied,
        ber=1e-6,          # placeholder; replaced by the measured BER
        overhead_bits=0.0,
    )


def _measure(cfg: FrontierConfig, band_lo: float, band_hi: float) -> tuple[DesignPoint, DesignBlerResult]:
    """Measure one band candidate; return ``(design point, measured result)``."""
    design = _build_design_point(cfg, band_lo, band_hi)
    result = measure_design_bler(
        design,
        mode="mfsk",
        num_symbols=NUM_SYMBOLS,
        rir=None,          # AWGN channel at cfg.snr_db (no RIR required)
        snr_db=cfg.snr_db,
        seed=SEED,
        band_lo=band_lo,
        band_hi=band_hi,
    )
    return design, result


def run(cfg: FrontierConfig) -> dict:
    """Measure every band candidate and extract the C2 frontier.

    Returns a dict with the per-band records, the Pareto front, the λ-sweep
    trace and the C2 verdict.
    """
    records: list[dict] = []
    # C2 cost measure (PLAN §2.6 / §9.4): perceptual annoyance when opted in via
    # ``cfg.use_perceptual_cost``, otherwise the legacy parametric dB audibility
    # cost (``cost_fn=None`` → ``aggregate_audibility_cost_db`` in the helpers).
    cost_fn = (
        make_perceptual_cost_fn(
            signal_type=cfg.perceptual_signal_type,
            noise_kind=cfg.perceptual_noise_kind,
        )
        if cfg.use_perceptual_cost
        else None
    )
    for cand in cfg.band_candidates:
        band_lo, band_hi = float(cand[0]), float(cand[1])
        try:
            design, result = _measure(cfg, band_lo, band_hi)
        except DesignInfeasibleError as exc:
            print(f"  [skip] band [{band_lo:g}, {band_hi:g}] infeasible: {exc}", file=sys.stderr)
            continue
        corrected = replace(design, ber=result.ber)
        records.append(
            {
                "band": (band_lo, band_hi),
                "result": result,
                "corrected": corrected,
                "goodput_bps": result.goodput_bps,
                "cost_db": result.audibility_cost_db,
                "cost_perceptual": cost_fn(corrected) if cost_fn is not None else None,
                "within_bler_budget": result.bler <= cfg.bler_max,
            }
        )

    if not records:
        raise RuntimeError("no feasible band candidate could be measured; cannot build the C2 frontier")

    points = [r["corrected"] for r in records]
    id_to_rec = {id(p): r for p, r in zip(points, records, strict=True)}

    pareto = nondominated(points, cost_fn=cost_fn)
    pareto_recs = [id_to_rec[id(p)] for p in pareto]
    trace = trace_optimum_over_lambda(points, cfg.lambda_grid, cost_fn=cost_fn)
    trace_recs = [id_to_rec[id(p)] for p in trace.frontier_points]
    verdict = c2_verdict(trace)

    return {
        "records": records,
        "pareto": pareto_recs,
        "trace": trace,
        "trace_recs": trace_recs,
        "verdict": verdict,
    }


def report(cfg: FrontierConfig, out: dict) -> None:
    """Print a human-readable report of the C2 frontier."""
    records = out["records"]
    trace = out["trace"]

    print("=" * 74)
    use_perceptual = bool(records) and records[0].get("cost_perceptual") is not None
    if use_perceptual:
        print("S17.2 — C2 FRONTIER:  goodput (bps)  vs.  perceptual annoyance (sone·penalty)")
    else:
        print("S17.2 — C2 FRONTIER:  goodput (bps)  vs.  audibility cost (dB)")
    print("=" * 74)

    bands = ", ".join(f"[{lo:g}, {hi:g}]" for lo, hi in cfg.band_candidates)
    print("config       : c2_frontier.yaml")
    print(f"  band candidates : {len(cfg.band_candidates)}  ({bands})")
    print(f"  spl_db = {cfg.spl_db:g} dB   snr_db = {cfg.snr_db:g} dB   bler_max = {cfg.bler_max:g}")
    print(f"  lambda_grid = {list(cfg.lambda_grid)}   aggregator = {cfg.aggregator}   room = {cfg.room}")
    print(f"  channel = AWGN @ snr_db (rir=None)   mode = MFSK   num_symbols = {NUM_SYMBOLS}   seed = {SEED}")
    print()

    print("per-band measured design points   (evidence: measured BER/BLER)")
    for r in records:
        lo, hi = r["band"]
        res = r["result"]
        budget = "ok" if r["within_bler_budget"] else "OVER"
        annoy = f"  annoy={r['cost_perceptual']:>7.2f}" if use_perceptual else ""
        print(
            f"  band [{lo:g}, {hi:g}] Hz   ber={res.ber:.3e}  bler={res.bler:.3e}  "
            f"goodput={r['goodput_bps']:>12,.0f} bps  cost={r['cost_db']:>7.2f} dB{annoy}  budget={budget}"
        )
    print()

    print(f"Pareto front  (maximise goodput, minimise cost) — {len(out['pareto'])} point(s)")
    for r in out["pareto"]:
        lo, hi = r["band"]
        annoy = f"   annoy = {r['cost_perceptual']:>7.2f}" if use_perceptual else ""
        print(f"  band [{lo:g}, {hi:g}] Hz   goodput = {r['goodput_bps']:>12,.0f} bps   cost = {r['cost_db']:>7.2f} dB{annoy}")
    print()

    print("lambda sweep  (trace_optimum_over_lambda over cfg.lambda_grid)")
    for lam, rec in zip(trace.lambdas, out["trace_recs"], strict=True):
        lo, hi = rec["band"]
        annoy = f"   annoy = {rec['cost_perceptual']:>7.2f}" if use_perceptual else ""
        print(
            f"  lambda={lam:>6g}   band [{lo:g}, {hi:g}] Hz   "
            f"goodput = {rec['goodput_bps']:>12,.0f} bps   cost = {rec['cost_db']:>7.2f} dB{annoy}"
        )
    print(f"  distinct frontier points (n_distinct) = {trace.n_distinct}")
    print()

    print("=" * 74)
    print(f"C2 VERDICT: {out['verdict']}")
    if out["verdict"] == "CORROBORATED":
        print("  the audibility cost discriminates across the frontier → the")
        print("  goodput-vs-cost trade-off is meaningful (C2 non-degenerate).")
    else:
        print("  all frontier points share one audibility cost → the metric")
        print("  fails to differentiate (C2 degenerate).")
    print("=" * 74)


def main() -> int:
    """Entry point: load config, compute the frontier, print the report."""
    cfg = load_frontier()
    out = run(cfg)
    report(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
