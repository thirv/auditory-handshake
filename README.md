# Auditory Handshake

**Auditory Handshake and Communication Protocol for Machine Listening** —
*an auditory control plane, not a new physical channel.*

When the RF stack (connection setup, association, re-association) is the latency
bottleneck for time-critical robot interaction, treat the acoustic medium as an
*alternative control plane*, benchmarked against the RF numbers it replaces.
Acoustic is not a faster link; it is a *different, measurable, physically
bounded* path with its own latency floor (C1), its own audibility/annoyance
optimum (C2), and its own rendezvous-plane framing (C3).

> **Status — pre-alpha, research-in-progress (M1–M4 + M6-TTCH simulation complete; M3 A/B caveats open).**
> In this repo today: an honest RF-baseline citation registry, a Shannon capacity
> ceiling, a free-space link-budget sweep, room presets with Sabine/Eyring RT60,
> RIR synthesis + channel metrics (RT60/`τ_rms`/`B50`/`B90`/DRR), noise
> generators, a motion/Doppler model, **two candidate modems (4-FSK and CSS)
> with a BER/BLER link over RIR + noise**, a **matched-filter preamble detector
> (M4) with TTFB detection-latency metrics** (`metrics/ttfb.py`; M4 exit
> criterion **p99 TTFB < 8 ms** — AWGN arm **PASS** p99 ≈ 1.25 ms, Channel arm
> **FAIL** p99 ≈ 293–378 ms, documented as RAKE/multi-path follow-up), and
> **an analytical slotted-ALOHA contention model for TTCH**
> (`sims/07_n_device_contention.py`; a deterministic T-vs-RT60 bound —
> model-derived, **not** a measurement) and **a measured two-way handshake
> round-trip driver** (`sims/09_two_way_handshake.py`): for each RT60 target it
> synthesizes a real RIR, measures RT60 back with the Schroeder integral,
> measures the reverb guard, and takes the true two-way RTT — then runs the C1
> falsifier. Its 5-room sweep gives Pearson r = **0.958** (t = 5.79), i.e. the
> RT60→RTT floor dependence is now **corroborated by a measured channel-model
> regression** rather than only an algebraic bound. **Every C1 result here is
> still simulation-level** (image-source room model; nothing is hardware-
> validated), and the sim_06 three-room *empirical* verdict is unchanged.
> Not
> here yet (and the point of the work): any coding/MCS, the handshake state
> machine, and an audibility metric. Note the CSS-vs-MFSK A/B in `sim_03` is
> **not yet a valid reverb-robustness result** — onset alignment via the
> M4 detector is not yet wired into that A/B path and the channel model has a
> fixed near-anechoic direct path (see the "Known limitations" section below).
> **Everything here is simulation; nothing is hardware-validated.** Do not read
> the milestones as achieved.

## Not novel in components (the honest verdict)

Per `PLAN.md` §9.3 and `RELATED_WORK.md` §9.3, the individual mechanisms are
*solved* (near-ultrasonic data exchange, dereverberation, acoustic encoders,
acoustic pairing + security, the underwater-UWA analogy). The defensible
contributions are *framing* and *consequence*, not the PHY:

- **C1** — the RT60-bounded handshake RTT floor and a *predictable* RTT-vs-RT60
  regression. *(best novelty candidate; a prediction, not yet a result)*
- **C2** — audibility/annoyance **in the objective function** (allowed to be a
  negative result).
- **C3** — the rendezvous-plane framing. *(framing, not novelty)*
- **C2 frontier (S17.2, 2026-10-14)** — `make sim-10` returns **CORROBORATED**: the perceptual audibility cost discriminates across the band frontier (2 distinct points; the optimum switches low↔high band at λ ≈ 5), so the audible-vs-near-ultrasonic band choice (D3) is now an explicit goodput↔annoyance trade-off surfaced by the metric, not an arbitrary pick. *Simulation-level: AWGN-only (BLER = 0), single SPL/room, `mfsk` perceptual signal — see the "Known limitations" section below.*

The *expected* reviewer reaction ("not novel in components") is documented as
such. The contribution is a measured, physically bounded control-plane benchmark
— not a new PHY.

## Start here (read in this order)

| File | What it is |
|---|---|
| [`PLAN.md`](PLAN.md) | The plan: claims (C1–C5), milestones, falsification criteria, working rules, risk register. The *stable* document. |
| [`RELATED_WORK.md`](RELATED_WORK.md) | Prior art, the citation-verification register, and the contribution sentence in §7. |
| [`SPEC.md`](SPEC.md) | Protocol-language skeleton (L0–L4) — headings only, no content yet. |
| `configs/` | YAML presets (devices, environment, scenario) — the only source of truth for parameters. |
| `src/auditory_handshake/` | Library: `config`, `metrics/capacity`, `acoustics/devices`, `models/rf_baseline`, `results`. |
| `sims/` | One file per experiment (`NN_name.py`), most run via `make sim-NN` — the `Makefile` exposes `sim-01` / `sim-02` / `sim-02b` / `sim-03` / `sim-04` / `sim-05` / `sim-07` / `sim-08` / `sim-09`. `06_ttch_c1_verification.py` has **no** `make` target — run it directly as `python sims/06_ttch_c1_verification.py`. `sim-09` is the measured C1 close-out (two-way handshake RTT vs measured RT60). |

## Working rules (from `PLAN.md` §1)

One task at a time · one task = one command · record the exact command + output ·
sub-second tests, fast iterations · **no hardware** (simulation
only). Reproducibility contract: `results/<exp>/<UTC-ts>/` with `config.json`,
`metrics.json` (seed + git SHA), and `figs/`.

## Known limitations (honest caveats)

This is a **simulation-only** codebase. Nothing here is hardware-validated, and
the headline C1 number is **model-derived** (two-way RTT = measured RT60 + a
~1.25 ms sync term), not an empirical RTT measurement. Known, documented
weaknesses:

- **Thin-RIR room model.** `compute_rir`'s image-source model produces a short,
  sparse tail (2–3 dominant paths), so `τ_rms`/`B50`/`B90` understate the true
  reverb. C1's *regression shape* (RT60 → RTT) holds by construction, but the
  room-metric magnitudes are approximate.
- **`sim_03` A/B is not a reverb-robustness result.** Onset alignment (the M4
  detector) is not yet wired into the A/B path, so "CSS beats MFSK" is a timing
  artifact, not a measured reverb win.
- **`fs` default footgun.** `compute_ber`/`compute_bler` silently default to
  `fs=16000`; pass `fs` explicitly for any other sample rate.
- **`sim_04` M4 gate: Channel arm fails.** p99 detection latency is ~1.25 ms on
  the AWGN arm (gate passes) but ~293–378 ms on the Channel arm (gate fails);
  RAKE/multi-path is the follow-up.

## Commands

```bash
conda env create -f environment.yml   # or: pip install -e ".[dev]"
make check                            # black --check + ruff + mypy + pytest
make test                             # pytest (tests/)
make sim-01                           # link-budget sweep  (-> results/)
```


## Licensing

This project is dual-licensed to separate the research content from the functional code:

* **Code**: All software code, scripts, and notebooks (`.py`, `.ipynb`, etc.) are licensed under the [MIT License](LICENSE).
* **Research & Documentation**: All written research, markdown files (`.md`), documentation, and data assets are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CONTENT).

If you use or adapt this work, please provide attribution to the original authors.


## Citation

If you use this research or code in your work, please cite it as follows:

```bibtex
@software{auditory-handshake2026,
  author       = {Hirvonen, Toni},
  title        = {Auditory Handshake and Communication Protocol for Machine Listening},
  month        = oct,
  year         = {2026},
  publisher    = {GitHub},
  version      = {0.1.0},
  url          = {https://github.com/thirv/auditory-handshake}
}
