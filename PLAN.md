# Auditory Handshake — Research & Development Plan

**Project:** `auditory-handshake` — an auditory (sound-based) information exchange
language for machine listening.
**Status:** Phase 1 (M1) complete (S0–S8 done); **M2 (S9–S11) complete** — S9.1 (room presets + Sabine/Eyring RT60, `acoustics/room.py`), S10 (RIR synthesis `acoustics/channel.py::compute_rir` + channel metrics `metrics/channel_metrics.py`), S11 (noise generators `acoustics/noise.py`, `sims/02_channel_characterization.py` + `make sim-02`, `configs/phy_constraints.yaml`); **M2 motion + M3 waveforms in** — S12.1 (motion/Doppler `acoustics/motion.py`), S12.2 (`sims/02b_motion_doppler.py` + `make sim-02b`), S13.1 (4-FSK `phy/mfsk.py`), S13.2 (CSS `phy/css.py`), S13.3 (BER/BLER link `metrics/ber.py` + `sims/03_css_vs_mfsk_ab.py` + `make sim-03`); all green (`make check` **233 passed**, `make types` clean). **Next — two known defects block valid results (see README → "Known limitations"):** (1) Phase A — `compute_rir` uses a 1 cm source–mic offset + fixed 0.25 s tail, so `τ_rms`/`B50`/`B90` measure geometry, not the room (fix: realistic src–mic distance, tail ≥ RT60, DRR sanity test, re-check `τ_rms ≈ 0.205·RT60`); (2) Phase B (M4) — no synchronisation/preamble stage, so `sim_03`'s CSS-beats-MFSK is a timing artifact, not a reverb result (add a matched-filter preamble detector + onset alignment, then re-run the A/B). **S9.2 (corridor, non-shoebox RIR path) still open.** C1 (the crux: TTCH vs RT60) is not yet measurable until Phases A+B land.
**Document license:** CC BY 4.0 (`LICENSE-CONTENT`). Code is MIT (`LICENSE`).
**Last updated:** 2026-10-06

---

## 0. Executive summary

Machines already talk over RF. This project asks whether **pressure vibrations in
air** — the channel that microphones and loudspeakers give every device for free —
can serve as a **low-latency, zero-association control channel** between nearby
machines, and how much information that channel can carry near its physical limit.

Two claims are developed and made falsifiable:

- **C1 — the handshake is bounded by the room, not the radio.** An acoustic link is
  forced half-duplex, so the room's reverberation decay sets a floor on
  round-trip handshake time. This *predicts which regimes win* (sub-meter
  proximity, car cabin, docking pad) and which cannot (reverberant halls).
- **C2 — audibility is a first-class cost.** Maximising rate subject to
  robustness and latency constraints, *jointly with* an acoustic-annoyance cost,
  yields a different optimum than optimising rate alone.

Plus the framing claim:

- **C3 — acoustic is a rendezvous/control plane, not a WiFi replacement.**
  Evaluated honestly against *connection establishment* (not against a
  monitor-mode RF receiver), acoustic wins on zero-association discovery.

- **C4 (candidate — *not yet a claim*) — spatial dimension.** A spatial knob
  (array index / bearing / room mode) extends the alphabet via spatial
  multiplexing, spatial/index modulation, SDMA selectivity, and
  bearing-as-semantic. **Deferred** behind the Phase 0/1/2 gates; in scope as
  cheap hooks only (D7). Inventory + falsification: `RELATED_WORK.md §8`.

The headline engineering targets are **~10² kbit/s** peak rate in favourable
conditions, **~6–8 ms time-to-first-bit**, and **~10–25 ms one-shot handshake**.

---

## 1. Motivation and the refined hypothesis

### 1.1 The original hypothesis, and why it needs sharpening

> **Naive form:** "Processing an auditory stimulus is faster than initialising a
> wireless connection, so machines should handshake acoustically."

This is directionally right but not falsifiable as stated, because a WiFi NIC in
**monitor mode** can receive a raw frame without associating, and a BLE radio
scanning for **advertisements** reports a packet in single-digit milliseconds.
Against a *receive-only* RF baseline, "listening is faster than radio init" is
false. The win is not the microphone; it is the **absence of a connection**.

### 1.2 The refined claim (this is the claim the project defends)

> The acoustic channel provides **zero-association, credential-free, broadcast,
> omni-directional discovery**, whose latency is bounded only by a preamble
> correlation window. It beats *connection-oriented* RF by one to two orders of
> magnitude, works when devices are network-isolated or air-gapped, and gives
> physically-bounded proximity as a security property. It is a **rendezvous and
> bootstrapping channel**, not a replacement for WiFi.

Complementary, and arguably the stronger motivation: acoustic works where RF is
**absent or prohibited** — Faraday enclosures, MRI suites, petrochemical and
mining zones, water-adjacent and metal-shielded equipment, sub-metre docking.

### 1.3 Baseline latency model (literature-derived, to be cited in M1)

Order-of-magnitude values; **all to be re-sourced and quoted with citations** in
`models/rf_baseline.py`. Marked `[V]` where we must verify.

| Path | TTFB / time-to-complete | Note |
|---|---|---|
| 802.11 open auth + assoc | ~30–100 ms `[V]` | no data yet |
| + WPA2 4-way + DHCP | ~150–600 ms `[V]` | connection-oriented |
| Wi-Fi, measured from user intent | 1–3 s `[V]` | UX reality |
| BLE connect attempt | ~1–3 s typical, 10 s+ worst `[V]` | scan interval/window |
| **BLE advertisement reception (no connect)** | ~3–30 ms `[V]` | **the honest hard baseline** |
| **WiFi monitor-mode frame RX (no assoc)** | ~1–10 ms `[V]` | **the honest hard baseline** |
| Acoustic preamble detect + header parse | **~6–8 ms (target)** | our design |
| Acoustic one-shot "shout" frame | **~10–25 ms (target)** | no reply |
| Acoustic 2-way handshake | **~30–120 ms (target)** | reverb-limited, see §2.4 |

Publishing the two bolded baselines is a deliberate credibility choice. If we
hide them, the work is dismissible.

### 1.4 Falsification criteria (written before implementation)

- **C3 dies** if a monitor-mode WiFi receiver, or a BLE scanner, delivers an
  equivalent payload at equivalent range within our acoustic **p95** latency.
- **C1 dies** if measured handshake round-trip time is statistically insensitive
  to `RT60` across the room presets (e.g. regression slope ≈ 0).
- **C2 dies** if the rate/robustness/latency Pareto frontier is degenerate — the
  same optimum is selected regardless of the weight on acoustic annoyance.

---

## 2. Physical-layer reality check

The design is decided here, not by taste. `[V]` = to be verified in simulation.

### 2.1 Bandwidth ceiling

Shannon: `C = B · log₂(1 + SNR)`.

| Usable band `B` | SNR | Capacity |
|---|---|---|
| 15 kHz (300 Hz–16 kHz) | 30 dB | ~150 kbit/s |
| 15 kHz | 20 dB | ~100 kbit/s |
| 15 kHz | 10 dB | ~52 kbit/s |
| 4 kHz (300–3400 Hz) | 20 dB | ~27 kbit/s |

Framing that survives scrutiny: human **semantic** speech is ~40–100 bit/s; a
speech codec is ~2–4 kbit/s. So the gap to human language is ~10³× at the
semantic layer and only ~10× at the waveform layer. What we change is the
**encoding**, aiming at channel capacity rather than at a speech codec.

Device reality (must be modelled, never assumed flat): small loudspeakers roll
off steeply below ~200–300 Hz; MEMS microphones typically stop near 16 kHz; both
are non-flat and non-linear at high SPL. **The usable band is a device-limited
sub-band**, so band selection is a first-class design variable.

### 2.2 Reverberation — the dominant obstacle (not noise)

Derived from a single-exponential decay:

- mean delay `τ̄ = RT60 / 6.9`
- RMS delay spread `τ_rms = √2 · τ̄ ≈ 0.205 · RT60` `[V]`
- therefore `RT60 = 0.5 s → τ_rms ≈ 100 ms` (diffuse upper bound; real RIRs less,
  because direct sound dominates early)

Design consequences:

- Coherence bandwidth `B_c ≈ 1 / (5 τ_rms)`: `τ_rms = 30 ms → ~6.7 kHz`;
  `100 ms → ~2 kHz`.
- OFDM requires `CP ≥ delay spread`. A 20 ms CP on a 25 ms symbol is **80 %
  overhead**. CP-OFDM is therefore *not* automatically correct here.
- Time-domain bound `R ≈ 1 / (10 τ_rms)`: `τ_rms = 30 ms → R ≈ 3.3 kBd`, so
  13 kbit/s at 4 bit/symbol, 26 kbit/s at 8 bit/symbol. Against a ~100 kbit/s
  Shannon bound that is a **~4× gap** — precisely the headroom this project
  attacks (channel-matched / time-reversal filtering, RAKE, DFE, channel
  sounding).

**This is the plan's central technical premise: reverberation, not the noise
floor, sets the achievable rate.**

### 2.3 Motion and clock drift

- Doppler `f_d = f·v/c`: at 10 kHz and 5 m/s → `f_d ≈ 146 Hz`, which is enormous
  against a 20 Hz subcarrier spacing. At 1 kHz → 14.6 Hz. A moving drone is a
  fast-fading, Doppler-heavy channel; pilots must be short and re-estimated often.
- Crystal offset (±20–50 ppm) gives only 0.2–0.5 Hz CFO at 10 kHz, and ~5 µs of
  drift over a 100 ms frame. **Doppler dominates; sample-clock drift is
  negligible at our frame durations.** `[V]`

### 2.4 Half-duplex is forced, and the room bounds the round trip

Speaker→own-mic coupling is 100+ dB, so **in-band full duplex is impossible**.
After a transmission ends, the room keeps ringing for ~`RT60`, so each turnaround
needs a guard on the order of the decay tail:

```
TTCH_RTT  ≥  2 · (T_preamble + T_frame + T_echo_tail) + T_proc
T_echo_tail ≈ k · RT60,  k ≈ 0.2–0.5 for a −30 dB decay threshold   [V]
```

This is claim **C1**'s quantitative form, and the reason to offer several frame
modes rather than one:

| Mode | Turnarounds | Latency | Gives up |
|---|---|---|---|
| Shout (unidirectional) | 0 | lowest | ACK, key agreement, reliability |
| 1.5-way (shout + NACK-only) | ≤1 | low | symmetric duplex semantics |
| 2-way handshake | 2 | ~2× turnaround | latency |
| 3-way + key agreement | 3 | highest | latency, for authenticity |

The **shout frame is the fast path** for the "drone arrives" scenario.

### 2.5 Noise

Small-quadrotor rotor noise is 70–90 dB SPL at 1 m, concentrated at the
blade-passing frequency and its harmonics (hundreds of Hz) with energy decaying
into the kHz range, plus broadband turbulence. HVAC/pink noise and speech add
more. Implication: **noise-adaptive spectral notching and per-sub-band MCS are
mandatory** — the achievable SNR can move by tens of dB between a quiet room and
a hovering drone.

### 2.6 Audibility and regulation (a real design axis)

Audible transmission in shared human space carries a social and sometimes legal
cost. Near-ultrasonic operation reduces annoyance but risks harm to children and
animals, hits aged-transducer rolloff, and is subject to SPL limits. Deployed
near-ultrasonic beacons generated genuine controversy — so **audibility belongs
in the objective function** (claim C2), not in a footnote.

---

## 3. The language: layers L0–L4

The project's product is a *language*, so it is specified as one, with a formal
grammar in `SPEC.md`:

- **L0 — Symbol layer.** MFSK tones, CSS chirps, OFDM subcarriers + QAM.
- **L1 — Frame layer.** `[SYNC | HEADER | PAYLOAD | CRC]`, bit-level grammar
  (ABNF-ish), CRC polynomials, MCS ladder table.
- **L2 — Session layer.** `HELLO → PROBE → OFFER → ACK → ASSOC` state machine,
  backoff, timeouts, collision handling.
- **L3 — Semantics layer.** Compact atomic vocabulary for embodied machines —
  capability tokens (`I_AM_DRONE`, `HAS_CAMERA`, `BATTERY_OK`), identity hashes,
  and **rendezvous descriptors** (WiFi SSID/PSK, IP, URI) so the acoustic link
  bootstraps a high-bandwidth RF link.
- **L4 — Discovery layer.** Broadcast beacons and paging sub-bands, trading
  latency against spectral efficiency.

---

## 4. System architecture

```
bits → framing/CRC → channel coding → interleaving → modulation
     → pulse shaping → loudspeaker response model
     → [ ROOM: CIR + reverberation + additive noise (ambient, rotor) ]
     → microphone + noise + AGC/limiter → sync (Zadoff-Chu / chirp matched filter)
     → CFO + channel estimation → equalisation → demod → decode → CRC
```

Candidates to be **tested, not assumed**: Zadoff-Chu/CAZAC preambles for
sample-accurate sync (single correlation peak ⇒ lowest detection latency and a
cheap false-alarm analysis); CSS/chirps for low-SNR robustness; MFSK with
differential detection for reverb robustness; OFDM-QAM for peak rate in benign
rooms; channel-matched/time-reversal filtering to close the §2.2 gap.

---

## 5. Simulation methodology

- **Engine:** `pyroomacoustics` (image-source for shoebox rooms, ray tracing for
  non-shoebox), plus our own device, noise and modem layers.
- **Room presets:** car cabin (~10 m³), small office (~40 m³), conference room
  (~150 m³), corridor (long, non-shoebox → ray tracing), small hall (~1000 m³),
  with frequency-dependent absorption materials.
- **Determinism:** every experiment is driven entirely by `configs/*.yaml` with
  recorded seeds; `results/<exp>/<UTC-timestamp>/` holds `config.json`
  (including git SHA), `metrics.json`, `figs/`, `tables/`.
- **Explicitly out of scope** (and stated as a validity threat): speaker
  non-linearity, true air absorption, and real AGC behaviour are *models*, not
  measurements, until an optional hardware spot-check in M7.

---

## 6. Environment

| Item | Value |
|---|---|
| Python | **3.12.14** (`>=3.12,<3.13`; conda env pins `python=3.12`) |
| `pyroomacoustics` | **0.10.1** (confirmed installed) |
| API note | module is `pyroomacoustics.signal` (lowercase); `Signal` does **not** exist |
| Dependency spec | `pyproject.toml` (primary) + `environment.yml` (conda fallback) |
| Install | conda env from `environment.yml`, then `pip install -e ".[dev]"` (verified at S0) |

Because 0.10.1 is much newer than the `0.6.x` lineage this plan was drafted
against, **step S1 is an API reconnaissance step** that records the verified API
surface in `docs/PYROOMACOUSTICS_API.md` before any simulation code is written,
and re-pins `pyroomacoustics` in `pyproject.toml`.

---

## 7. Work plan

### 7.1 Near-term increments (local-model-friendly: small steps, one command each)

| Step | Deliverable | Checkpoint |
|---|---|---|
| **S0** ✅ | `pyproject.toml`, `environment.yml`, `.gitignore`, minimal package | **done — env installed, pra 0.10.1** |
| **S1** ✅ | API reconnaissance → `tools/probe_pra_api.py`, `docs/PYROOMACOUSTICS_API.md`, re-pin pyproject | **done — `python tools/probe_pra_api.py` runs clean; API doc reviewed** |
| **S2** ✅ | Package skeleton, `Makefile`, `results.py`, `configs/*.yaml` | **done — `make lint && make test` green** |
| **S3** ✅ | `metrics/capacity.py` + unit tests | **done — capacity table printed** |
| **S4** ✅ | `acoustics/devices.py` (speaker/mic band models) | **done — `make test` green** |
| **S5** ✅ | `models/rf_baseline.py` + cited baseline table | **done (S5.2 figure-sourcing tail open, non-blocking)** |
| **S6** ✅ | `sims/01_link_budget.py` | **done — `make sim-01` → figures + `metrics.json`** |
| **S7** ✅ | **`RELATED_WORK.md` + contribution sentence decision gate** | **done — gate closed (D4); 4 arXiv IDs `[PENDING]` re-verify, non-blocking** |
| **S8** ✅ | `README.md` (short description) ✓, `SPEC.md` skeleton ✓, `SPEC.md` finalised (L0–L4 content) at M9 | **docs done; SPEC.md content + open figures remain** |
| S9 | `acoustics/room.py` presets | `make test` |
| S10 | `acoustics/channel.py` (RT60, τ_rms, B_50) + tests | `make test` |
| S11 | `sims/02_channel_characterization.py` | `make sim-02` → `phy_constraints.md/.yaml` |
| S12 | `sims/02b_motion_doppler.py` | `make sim-02b` |

### 7.2 Milestone map

| M | Deliverable | Exit criterion |
|---|---|---|
| **M0** | Scaffold, env, spec skeleton | `make test` green; pyroomacoustics imports |
| **M1** | Link budget, capacity ceiling, honest RF baseline | documented ceiling + falsification criteria |
| **M2** | Channel characterisation across room presets | measured τ_rms vs prediction; `configs/phy_constraints.yaml` published |
| M3 | Waveform A/B: MFSK vs CSS vs OFDM-QAM vs channel-matched | BER/BLER curves; winner per RT60 regime; reverb gap to Shannon |
| M4 | Sync & framing: preamble detect latency, false-alarm rate | p99 detect < 8 ms at target SNR/FAR — **detection latency (TTFB) measured & passing on the AWGN arm (p99 ≈ 1.25 ms, see `metrics/ttfb.py`); Channel arm still fails (p99 ≈ 293–378 ms, RAKE/multi-path follow-up)** |
| M5 | Coding, MCS ladder, link adaptation, HARQ | goodput within ~3–4× of Shannon at high SNR |
| M6 | Handshake FSM, backoff, N-device contention | TTCH CDF vs device count; **C1** confirmed or killed |
| M7 | Robustness: rotor noise, AGC non-linearity, motion | graceful degradation; notching recovers ≥10 dB |
| M8 | Security: eavesdrop, spoof, jam, channel-reciprocity keys | attacker-distance / decorrelation analysis |
| M9 | Pareto frontier, spec freeze, write-up | frontier figure including audibility; `SPEC.md` v1.0 — **S17.1 done** (perceptual annoyance metric + non-negativity regression, `make check` 458 green, 2026-10-13); S17.2 `FrontierConfig` landed in `config.py`; **open: sim-10 wiring + commit** |

---

## 8. Metrics

BER · BLER · goodput · spectral efficiency vs Shannon · **TTFB** · **TTCH
(p50/p95/p99)** · false-alarm rate · discovery probability vs range ·
energy per delivered bit · an **audibility score**.

Success thresholds for M1/M2: capacity ceiling documented and defended;
τ_rms prediction quantified as a measured band, not forced to agree; a
`phy_constraints.yaml` that M3 can design against without further assumptions.

---

## 9. Preliminary novelty analysis

### 9.1 Method and limitation

A preliminary search of **arXiv only** (API metadata queries; a Semantic Scholar
query was rate-limited). This is **not** a literature review. Critically, the
relevant corpus is largely *not* on arXiv — room acoustics and acoustic
communication publish in **JASA, IEEE TASLP/TCOM/ICASSP, Applied Acoustics**, and
the **underwater acoustics** literature (e.g. Stojanovic) — so a zero-hit query
is **not** evidence of novelty. The claims in §1.2 are provisional until S7.

### 9.2 Prior art that directly overlaps

| Prior work | Overlap |
|---|---|
| Tabak, Lin & Singer, *High Data Rate Near-Ultrasonic Communication with Consumer Devices*, EUSIPCO 2021 (arXiv:2103.11261) — **4 kbps over 5 m**, 18–20 kHz, phase-coherent adaptive equalisation; explicitly identifies indoor near-ultrasonic ≈ underwater acoustic channels | This **is** a high-bandwidth auditory exchange language, and already contains the underwater-transfer insight. Framed around device pairing and credential exchange. |
| Putz, Álvarez & Classen, *Acoustic Integrity Codes: Secure Device Pairing Using Short-Range Acoustic Communication*, ACM WiSec 2020 (arXiv:2005.08572) — 100 bps @ 14 dB SNR, BER < 0.1 %, Android implementation, anti-overshadowing | Occupies "acoustic pairing + security layer". |
| Matt & Stowell, *Estimating & Mitigating the Impact of Acoustic Environments on Machine-to-Machine Signalling*, 2019 (arXiv:1908.04672) — reverberation parameter estimation + dereverberation for a real M2M decoder, audible and ultrasonic | Almost exactly our **M2**, including the finding that speech dereverberation may not generalise to non-speech signals. |
| Murano et al., *Evaluation of Zadoff-Chu, Kasami and Chirp based encoding schemes for Acoustic Local Positioning Systems*, IEEE TIM 2020 (arXiv:2402.02400) — multipath, Doppler, multiple-access interference, near-far | Our planned M4 waveform space is already benchmarked. |
| Shen et al., *A Hybrid Advertising Mode for Device Discovery in BLE Networks*, 2023 (arXiv:2312.15650) — analytic discovery-latency models | Confirms RF discovery latency is a studied, quantified field; our C3 baseline is contested territory. |
| Hogg & Freitas, *Acoustic Communication for Medical Nanorobots*, Nano Comm. Networks 2012 (arXiv:1202.0568); Lieser & Schwanecke, RO-MAN 2024 (arXiv:2312.17668, rotor noise masks acoustic signalling) | Acoustic robot-swarm comms, and the rotor-noise obstruction, are on record. |
| Deployed systems: Chirp (2011), Google Tone (2015), near-ultrasonic TV→phone beacons, acoustic IoT provisioning; NTT "acoustic OFDM" (~2007–09) | "Machines exchanging data over sound" is a shipped product category. |

### 9.3 Verdict

**Not novel in its components.** The modulations, the reverb mitigation, the
pairing use case, and even the underwater-acoustics analogy are published. The
dismissal to expect:

> *"This is data-over-sound — acoustic OFDM since ~2009, Chirp since 2011, NUSC
> at 4 kbps since 2021 — repackaged with a latency framing."*

Three places a defensible contribution can still live:

1. **Framing and benchmarking (moderate; solid systems contribution).** Prior
   work treats acoustic transfer as a *security* channel, a *positioning* aid, or
   a raw-throughput exercise. None frames it as a **latency-optimal control plane
   benchmarked against RF connection establishment** with TTFB/TTCH distributions
   and published monitor-mode/BLE-adv baselines.
2. **Claim C1, the RT60-bounded handshake round trip (best novelty candidate).**
   No located work states that forced half-duplex plus reverberation decay imposes
   a *protocol-level floor* on handshake RTT. It is crisp, cheap to test, and
   predicts the winning regime instead of asserting "sound is faster".
3. **Claim C2, audibility in the objective (freshest; may yield a negative
   result).** Jointly optimising rate × robustness × latency against an acoustic
   annoyance cost is not represented in the located literature, and the socially
   interesting outcome may be *what should not be deployed*.

### 9.4 Consequence for the plan

- **S7 (`RELATED_WORK.md`) is a decision gate**, not documentation: we write the
  single contribution sentence *before* writing the README, and we state which
  claim we lead with.
- Realistic assessment: a **strong systems contribution with two modest novel
  claims**, plus a first-rate learning and reproduction vehicle. Not a new field.
- Likely venues favour systems/robotics framing (MobiCom/MobiSys, PerCom,
  ICRA/IROS, RO-MAN) over signal-processing venues, where the PHY alone would be
  judged derivative.

---

## 10. Threats to validity

1. Naive-hypothesis framing without RF admission baselines → mitigated by §1.2–1.4.
2. `pyroomacoustics` is a geometric model; it does not capture true air
   absorption, transducer non-linearity, or real AGC → modelled explicitly and
   flagged, optional hardware spot-check later.
3. Prior art may already contain C1/C2 → S7 gate, before claims are published.
4. Simulated-only results → stated as a limitation until M7.
5. Scope creep across PHY + MAC + security → strict per-milestone exit criteria;
   simulation-only until M7.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **`pyroomacoustics` 0.10.1 API drift (realised)** | S1 API probe; all engine use behind our own `acoustics/` layer |
| Reverb gap (~4× below Shannon) hard to close | channel-matched filtering; M3 measures it before we commit |
| Over-claiming novelty | S7 decision gate; honest baselines published |
| Local-model context loss between steps | small steps, one command per checkpoint |

## 12. Reproducibility policy

All parameters in `configs/*.yaml`; seeds and git SHA recorded in every
`metrics.json`; every figure regenerable by a single `make` target; `make test`
sub-second so validation is always cheap.

## 13. Open questions / decisions log

| # | Question | Status |
|---|---|---|
| D1 | Environment-first ordering | **Decided** — API drift confirmed material (0.10.1) |
| D2 | Honest RF baselines published | **Decided** — yes, credibility |
| D3 | Band choice: audible vs near-ultrasonic | Open — becomes part of the C2 frontier |
| D4 | Which claim leads the README | **Closed** (S7.2, 2026-09-28) — **systems contribution** leads (latency-optimal control plane benchmarked vs RF connection establishment); C1 crux, C2 freshest, C3 scope |
| D5 | Hardware validation in scope? | Open — optional M7 |
| D6 | Learned receivers (neural demod) vs classical | Open — optional M9 |
| D7 | Spatial scope: hooks only vs full Phase C | **Decided (2026-09-25)** — hooks only, Phase C deferred; candidate C4 inventory + falsification (`spatial_dof()`) in `RELATED_WORK.md §8` |
| D8 | Dual-use / opportunistic acoustic channels (hybrid host-audio; noise-as-carrier) in scope? | **Open (2026-09-28)** — recorded as a candidate extension, deferred, hooks only; candidate C5 inventory + falsification criteria in `RELATED_WORK.md §9`; not a commitment |
| D9 | Ultrasonic effect on C1 (TTCH floor) | **Open (2026-10-14)** — only route is air absorption / band-dependent RT60; current RIR is single-band (frequency-flat) so C1's floor is band-independent. Deferred — C2 does not depend on it. |
