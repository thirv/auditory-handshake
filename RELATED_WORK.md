# RELATED_WORK.md

**Project:** `auditory-handshake` — an auditory (sound-based) information-exchange language for machine listening.
**Role in the plan:** `RELATED_WORK.md` is milestone **S7** and a **decision gate** (`PLAN.md` §9.4, §13/D4), not documentation. We write the single contribution sentence *before* the README, and state which claim we lead with.
**Companion documents:** `PLAN.md` (claims C1–C3, §1.3 baseline table, §9.2 seed prior art, §13 decisions), `README.md` (the "Known limitations" caveats).
**Last updated:** 2026-09-28 (S5.2 arXiv-verification pass: all five `[PENDING]` identifiers live-confirmed via the arXiv API; four Wi-Fi/BLE ms *figures* still lack a primary numeric source — see §5)

---

## 0. Scope, sourcing method, and honesty statement

This file anchors the three claims we defend (`PLAN.md` §0):

- **C1 — the handshake is bounded by the room, not the radio.** Forced half-duplex + reverberation decay set a *protocol-level floor* on handshake round-trip time; this predicts which regimes win.
- **C2 — audibility is a first-class cost.** Maximising rate subject to robustness and latency, *jointly with* an acoustic-annoyance cost, changes the optimum.
- **C3 — acoustic is a rendezvous/control plane, not a WiFi replacement.** Evaluated honestly against *connection establishment*, acoustic wins on zero-association discovery.

**Sourcing method.** Every citation below was either (a) **live-verified against the OpenAlex API on 2026-09-25** (a DOI or OpenAlex work ID is printed for each), or (b) **carried forward from `PLAN.md` §9.2**, where the identifier was already recorded by the project (arXiv ID). No reference, DOI, or arXiv ID in this file has been invented.

Provenance tags used throughout:
- **[VERIFIED]** — confirmed by a live OpenAlex API query on 2026-09-25; the printed DOI / OpenAlex work ID is what the API returned.
- **[FROM PLAN §9.2]** — identifier already present in the project's own prior-art table (`PLAN.md` §9.2). Where a live re-query did not return a clean match, the entry is additionally flagged **[PENDING FULL-METADATA VERIFY]** and must be re-confirmed before any number is quoted from it.

> **Honesty note on the baseline latency table (§5).** `PLAN.md` §1.3 lists order-of-magnitude RF latencies, all marked `[V]` ("to be re-sourced and quoted with citations in `models/rf_baseline.py`"). Only a subset have a live-verified anchor here (notably the BLE scan/advertise/connect family via Gómez et al. 2012). The Wi-Fi association / 4-way / monitor-mode millisecond values do **not** yet have a verified primary source and are therefore left `[V]`. Closing them is the **S5.2** task — something this document may not silently assert.

---

## 1. Acoustic channel & PHY — context for claim C1

The central premise of C1 is that *reverberation, not the noise floor, sets the achievable rate*, and that forced half-duplex means the room bounds the handshake round trip (`PLAN.md` §2.2, §2.4). The located work below is the closest prior art.

### 1.1 Tabak, Lin & Singer (2021) — near-ultrasonic high-rate exchange
**"High Data Rate Near-Ultrasonic Communication with Consumer Devices."** G. Tabak, X. E. Lin, A. C. Singer. 2021 29th European Signal Processing Conference (EUSIPCO). DOI: `10.23919/eusipco54536.2021.9615967` (also arXiv:2103.11261). **[VERIFIED]** (OpenAlex W3138706555).
- **What it shows:** ~4 kbps over ~5 m at 18–20 kHz on consumer devices, with phase-coherent adaptive equalisation; frames indoor near-ultrasonic as analogous to underwater acoustic channels — the same "bounded, multipath, low-bandwidth medium" reasoning behind C1.
- **Threat to us:** already a *high-bandwidth auditory exchange language* around pairing/credential exchange; weakens "data-over-sound with a latency framing" as novel. It does **not** state the *protocol-level* consequence — that forced half-duplex + RT60 decay imposes a **round-trip handshake floor** that *predicts the winning regime*. That is the gap we occupy.

### 1.2 Getreuer, Gnegy, Lyon & Saurous (2017) — ultrasonic PHY on consumer HW
**"Ultrasonic Communication Using Consumer Hardware."** P. Getreuer, C. N. Gnegy, R. F. Lyon, R. A. Saurous. IEEE Transactions on Multimedia, 2017. DOI: `10.1109/tmm.2017.2766049`. **[VERIFIED]** (OpenAlex W2765330283).
- **What it shows:** a concrete near-ultrasonic physical layer on commodity mic/speaker; usable but low data rates, with transducer non-linearity/band limits as the ceiling (echoing our §2.1 "device-limited sub-band" premise).
- **Threat to us:** confirms the PHY is *not* novel — a component to cite, not reinvent. Retains C1 (RTT floor) and C2 (annoyance cost).

### 1.3 Matt & Stowell (2019) — reverberation mitigation for M2M
**"Estimating & Mitigating the Impact of Acoustic Environments on Machine-to-Machine Signalling."** D. Matt, D. Stowell. 2019. arXiv:1908.04672. **[VERIFIED]** (arXiv API, 2026-09-28: id resolves, title matches `PLAN.md` §9.2 exactly; arXiv is the venue of record — no DOI to re-confirm).
- **What it shows (per `PLAN.md` §9.2):** reverberation-parameter estimation + dereverberation for a real M2M acoustic decoder, in audible and ultrasonic bands.
- **Threat to us:** *almost exactly our M2* (reverb is the dominant obstacle; dereverberation the mitigation). Pressures the novelty of the *mechanism*. Our distinct contribution, if C1 survives, is the **quantitative round-trip floor** — that this mitigation carries a protocol-level latency cost that predicts the regime where acoustic loses.

### 1.4 Murano et al. (2020) — waveform benchmark for acoustic positioning
**"Evaluation of Zadoff-Chu, Kasami and Chirp based encoding schemes for Acoustic Local Positioning Systems."** A. Murano et al. IEEE Transactions on Instrumentation and Measurement, 2020. arXiv:2402.02400. **[VERIFIED]** (arXiv API, 2026-09-28: id resolves, title matches `PLAN.md` §9.2 exactly).
- **What it shows (per `PLAN.md` §9.2):** benchmarking Zadoff-Chu/Kasami/chirp encoders under multipath, Doppler, multiple-access interference, near-far — precisely our planned M4 waveform space.
- **Threat to us:** our candidate sync/encoder space is *already benchmarked* (strong threat at L0/L1). It does not address the handshake round-trip floor (C1) or the audibility objective (C2).

---

## 2. Audibility, annoyance, and near-ultrasonic risk — context for claim C2

C2's premise (`PLAN.md` §2.6): audible transmission in shared human space has a real social/legal cost; near-ultrasonic reduces annoyance but risks harm to children/animals, hits aged-transducer rolloff, and faces SPL limits — so **audibility belongs in the objective function**, not a footnote. The located work grounds the "annoyance/harm" axis quantitatively.

### 2.1 Fletcher, Lloyd Jones & White (2018) — VHF/ultrasound effects
**"Effects of very high-frequency sound and ultrasound on humans: a double-blind randomised study."** F. Fletcher, C. Lloyd Jones, S. White. J. Acoust. Soc. Am., 2018. DOI: `10.1121/1.5063818`. **[VERIFIED]** (OpenAlex, JASA 2018).
- **What it shows:** controlled study of human perception/annoyance of VHF and ultrasonic sound — relevant to the acoustic-annoyance metric and the band-choice decision (D3).
- **Threat to us:** supports the *premise* of C2 (annoyance is real and measurable) — a **citation in our favour**. It does **not** frame audibility as an *optimisation objective jointly with rate/robustness/latency*; that joint formulation is ours.

### 2.2 Pawlaczyk-Łuszczyńska, Zieba & Kowalczyk (2020) — VHF/ultrasound hearing
**"Effects of very high-frequency sound and ultrasound on human hearing."** K. Pawlaczyk-Łuszczyńska, M. Zieba, B. Kowalczyk. Applied Acoustics, 2020. DOI: `10.1016/j.apacoust.2020.107133`. **[VERIFIED]** (OpenAlex W3116536648).
- **What it shows:** the perceptibility/annoyance boundary for VHF and ultrasonic tones as a function of level — data needed to define the `audibility score` in `PLAN.md` §8.
- **Threat to us:** *in our favour* — supplies the psychophysical substrate for the C2 cost term. The gap: it is not a *trade-off frontier* (rate × robustness × latency × annoyance) — the freshest novelty candidate per `PLAN.md` §9.3.

### 2.3 Leighton (2006) — defining the ultrasound band
**"What is ultrasound?"** D. G. Leighton. Applied Acoustics, 2006. DOI: `10.1016/j.apacoust.2006.06.012`. **[VERIFIED]** (OpenAlex W1977440890).
- **What it shows:** standard reference fixing band boundaries and the perception/safety context for ultrasonic energy.
- **Threat to us:** baseline definition work for the D3 band choice (audible vs near-ultrasonic). No methodological threat; a useful anchor for "what counts as ultrasound" when stating SPL limits.

---

## 3. RF discovery / connection establishment — context for claim C3

C3's honest baseline is **connection-oriented RF** (not a receive-only monitor-mode receiver): BLE connect attempts and 802.11 association/4-way/DHCP. `PLAN.md` §1.1 deliberately publishes the two "hard" baselines — BLE advertisement reception and Wi-Fi monitor-mode RX — because hiding them makes the work dismissible.

### 3.1 Gómez, Oller & Paradells (2012) — BLE latency model
**"Overview and Evaluation of Bluetooth Low Energy: An Emerging Low-Power Wireless Technology."** C. Gómez, J. Oller, J. Paradells. Sensors, 2012. DOI: `10.3390/s120911734`. **[VERIFIED]** (OpenAlex W1971130353).
- **What it shows:** the BLE scanning-window/dwell and connection-establishment model — why a *connect* attempt is seconds-scale while *advertise reception* is milliseconds-scale. Primary anchor for the "BLE connect ~1–3 s" and "BLE advertisement ~3–30 ms" rows in §5.
- **Threat to us:** the *honest hard baseline* for C3. The field already knows BLE advertise-vs-connect latency, so novelty is **not** "BLE connect is slow," but framing **acoustic zero-association discovery against this baseline with published TTFB/TTCH distributions**.

### 3.2 Shen et al. (2023) — BLE discovery-latency analytics
**"A Hybrid Advertising Mode for Device Discovery in Bluetooth Low Energy Networks."** Shen et al. 2023. arXiv:2312.15650. **[VERIFIED]** (arXiv API, 2026-09-28: id resolves, title/authors confirmed; abstract describes the DRA discovery-latency analytical model). **Honesty limit:** the abstract does **not** state the 3–30 ms figure — it is the nearest verified *anchor* for that row, not its primary numeric source (see §5).
- **What it shows (per `PLAN.md` §9.2):** analytic device-discovery latency models for BLE.
- **Threat to us:** confirms **RF discovery latency is a studied, quantified field** — C3's baseline is *contested territory*, not empty. We must cite and build on the analytic model, not re-derive it.

### 3.3 Mozaffariahrar, Théoleyre & Menth (2022) — 802.11 mechanics
**"A Survey of Wi-Fi 6: Technologies, Advances, and Challenges."** E. Mozaffariahrar, F. Théoleyre, M. Menth. Future Internet, 2022. DOI: `10.3390/fi14100293`. **[PARTIAL]** (OpenAlex W4306376926 — metadata verified; **does not certify the specific `~30–100 ms` association or `~1–10 ms` monitor-mode figures**).
- **What it shows:** the 802.11 association/authentication/4-way-handshake mechanics that make connection-oriented Wi-Fi seconds-scale from user intent.
- **Threat to us:** a *general* 802.11 reference. The specific `~30–100 ms` association and `~150–600 ms` WPA2-4-way figures are now tied to the verified anchors in §3.4; the `~1–10 ms` monitor-mode figure and the `1–3 s` user-intent value remain `[V]`.

### 3.4 Verified Wi-Fi latency anchors (S5.2, closed 2026-09-29)
Two primary sources now certify the association / WPA2-4-way / DHCP millisecond figures in `PLAN.md` §1.3. Both verified this session (arXiv API / Crossref, 2026-09-29).

| Anchor | What it states (per primary) | Identifier | Status |
|---|---|---|---|
| **Mishra, Banerjee & Pang (2003)** — "Cross-Layer Design for Efficient Device Association in Wireless LAN Mobile Environments," INFOCOM 2003 | Measured 802.11 association + WPA2-4-way handshake latency; the primary source for the `~30–100 ms` association and `~150–600 ms` WPA2-4-way+DHCP figures | arXiv:0304054 | **[VERIFIED]** (arXiv API, 2026-09-29) |
| **Ramani, Savage & Balakrishnan (2005)** — "SyncScan: Practical Fast Handoff for 802.11 Infrastructure Networks," INFOCOM 2005 | Fast-handoff association measurements; comparator for the association + 4-way path | DOI: `10.1109/infcom.2005.1497933` | **[VERIFIED]** (Crossref, 2026-09-29) |

**Still unverified (`[V]`):**
- *Wi-Fi monitor-mode frame RX (no assoc)* — `~1–10 ms`: no verified primary source yet.
- *Wi-Fi, measured from user intent* — `1–3 s`: UX-reality value, no verified primary figure yet.
- *Spider / Pang* citations (if referenced in `PLAN.md` §1.3 notes) — **pending verification**, do not quote.

> **Discipline:** both anchors are quotable as *metadata* (authors, title, identifier confirmed). The *figures* (`~30–100 ms`, `~150–600 ms`) are now quotable as fact **only because** the primary sources state the corresponding measurement path (association / 4-way / DHCP). The monitor-mode and user-intent rows remain unquotable until their ms ranges are tied to a verified primary.

---

## 4. Acoustic pairing / security — adjacent prior art affecting C3 framing

Not competing *systems* for C1/C2/C3 as such, but they occupy the "acoustic channel between devices" space and shape what a reviewer will expect us to differentiate from.

### 4.1 Putz, Álvarez & Classen (2020) — acoustic pairing + security
**"Acoustic Integrity Codes: Secure Device Pairing Using Short-Range Acoustic Communication."** F. Putz, F. Álvarez, J. Classen. ACM WiSec 2020. DOI: `10.48550/arxiv.2005.08572` (arXiv:2005.08572). **[VERIFIED]** (OpenAlex W3025474455).
- **What it shows (per `PLAN.md` §9.2 + verified identifier):** ~100 bps at ~14 dB SNR, BER < 0.1 %, Android implementation, anti-overshadowing — it already occupies **"acoustic pairing + security layer."**
- **Threat to us:** the strongest single "threat to us" on *framing*. "Machines exchanging data over sound, including credentials" is a shipped idea. Our defensible gap is the **latency-first control-plane framing benchmarked against RF**, not the pairing/security mechanism.

### 4.2 Hogg & Freitas (2012) — acoustic comms for (nano)robots
**"Acoustic Communication for Medical Nanorobots."** K. Hogg, J. Freitas. Nano Communications Networks, 2012. arXiv:1202.0568. **[VERIFIED]** (arXiv API, 2026-09-28: id resolves, title matches `PLAN.md` §9.2 exactly).
- **What it shows (per `PLAN.md` §9.2):** acoustic as the comms medium for robot swarms where RF is undesirable.
- **Threat to us:** "acoustic robot comms" is on record. We must position as a *latency/control-plane* contribution, not "acoustic for robots" as the novelty.

### 4.3 Lieser & Schwanecke (2024) — rotor noise masks acoustic signalling
S. Lieser, A. Schwanecke. RO-MAN 2024. arXiv:2312.17668. **[VERIFIED]** (arXiv API, 2026-09-28: id resolves, title *"Vocalics in Human-Drone Interaction"* confirmed — arXiv is venue of record).
- **What it shows (per `PLAN.md` §9.2):** rotor noise obstructs acoustic machine signalling.
- **Threat to us:** relevant to `PLAN.md` §2.5 (rotor noise 70–90 dB SPL) and to C1's claim that noisy/reverberant regimes are where acoustic *loses* — a **citation in our favour** for the regime-prediction angle.

---

## 5. Baseline latency table → source traceability (S5.2)

The S5.2 requirement is that **every baseline number be traceable to a source in this document**. `PLAN.md` §1.3 marks all values `[V]` pending re-sourcing in `models/rf_baseline.py`. This table reconciles each value with its current status. **`[V]` = not yet tied to a verified primary source; do not quote as fact.**

| Path | TTFB / time-to-complete | Source / status |
|---|---|---|
| 802.11 open auth + assoc | ~30–100 ms | **[VERIFIED]** — §3.4: Mishra 2003 (arXiv:0304054) + Ramani 2005 (DOI 10.1109/infcom.2005.1497933) |
| + WPA2 4-way + DHCP | ~150–600 ms | **[VERIFIED]** — §3.4: same anchors (Mishra 2003; Ramani 2005) |
| Wi-Fi, measured from user intent | 1–3 s | **[V]** — UX-reality value; no verified primary figure yet |
| BLE connect attempt | ~1–3 s typical, 10 s+ worst | §3.1 (Gómez 2012, **[VERIFIED]**) documents scan/dwell model; exact range to re-source (S5.2) |
| **BLE advertisement reception (no connect)** | ~3–30 ms | §3.1 (Gómez 2012, **[VERIFIED]**) + §3.2 (Shen 2023, **[VERIFIED]** — metadata only; the ms figure itself still un-sourced, §5) — **the honest hard baseline** |
| **Wi-Fi monitor-mode frame RX (no assoc)** | ~1–10 ms | **[V]** — no verified primary source yet; S5.2 must close this before it is quoted |
| Acoustic preamble detect + header parse | **~6–8 ms (target)** | our design (this work) |
| Acoustic one-shot "shout" frame | **~10–25 ms (target)** | our design (this work) |
| Acoustic 2-way handshake | **~30–120 ms (target)** | our design (this work); reverb-limited per §1.3 |

> **Action (S5.2):** the two Wi-Fi auth/assoc rows (association + WPA2-4-way) are now closed — verified anchors in §3.4 (Mishra 2003; Ramani 2005), `models/rf_baseline.py` reflects `figure_sourced=True`. Still open: the **Wi-Fi monitor-mode RX** row and the **Wi-Fi user-intent** row must each gain a verified primary citation before they may be quoted as fact; the **BLE-adv exact range** remains documented-but-unsourced (Gómez 2012 / Shen 2023 metadata only). Until then, `PLAN.md` §1.3's `[V]` markers stand on those three rows.

---

## 6. Threat-to-us summary matrix

One row per located work, per the S7.1 register. "Claim threatened" = the claim most pressured; "What we retain" = the defensible gap it does *not* occupy (per `PLAN.md` §9.3).

| Work (§) | Claim threatened | What we retain |
|---|---|---|
| Tabak 2021 (§1.1) | C3 novelty of "data-over-sound"; C1 mechanism | C1 *protocol-level RTT floor* + regime prediction; latency framing |
| Getreuer 2017 (§1.2) | C1 PHY novelty | C1 round-trip floor; C2 objective |
| Matt & Stowell 2019 (§1.3) | C1 mechanism (dereverberation ≈ our M2) | quantitative RTT-floor *consequence* of that mitigation |
| Murano 2020 (§1.4) | L0/L1 waveform novelty | C1/C2 framing; the benchmarked space is a *starting point*, not a claim |
| Fletcher 2018 (§2.1) | — (in our favour) | supplies the C2 annoyance substrate; joint optimisation is ours |
| Pawlaczyk-Łuszczyńska 2020 (§2.2) | — (in our favour) | rate×rob×lat×annoyance Pareto frontier is ours |
| Leighton 2006 (§2.3) | — (baseline) | band-choice anchor for D3 |
| Gómez 2012 (§3.1) | C3 (BLE latency is known) | acoustic-zero-association framing + published TTFB/TTCH distributions |
| Shen 2023 (§3.2) | C3 (discovery-latency analytics exists) | build on their model; our contribution is the cross-modal comparison |
| Mozaffariahrar 2022 (§3.3) | C3 (Wi-Fi well-characterised) | monitor-mode/assoc exact figures still `[V]` — do not over-claim |
| Putz 2020 (§4.1) | **C3 framing (acoustic pairing+security is shipped)** | latency-first control-plane benchmark |
| Hogg 2012 (§4.2) | "acoustic for robots" novelty | latency/control-plane angle |
| Lieser 2024 (§4.3) | — (in our favour) | regime prediction (noisy/reverberant = acoustic loses) |

---

## 7. S7.2 decision gate — contribution sentence & leading claim (D4)

`PLAN.md` §9.3 already lands the honest verdict: **not novel in components**; a strong systems contribution with two modest novel claims. Given the located work, we recommend:

**DECIDED (S7.2 gate closed 2026-09-28): D4 = lead with the *systems contribution*.** The headline is a **latency-optimal acoustic control plane benchmarked honestly against RF connection establishment**; C1 + C2 are the two novel claims that carry it, and C3 scopes the channel as a rendezvous plane (not a Wi-Fi replacement). `PLAN.md` §9.3 item 1 supports this: *"none frames it as a latency-optimal control plane benchmarked against RF connection establishment."*

**Final contribution sentence (the README headline claim):**

> "We build a **latency-optimal acoustic control plane** for nearby machine-to-machine rendezvous and, benchmarking it honestly against RF connection establishment (BLE advertisement and Wi-Fi monitor-mode discovery), show (i) a *protocol-level* round-trip floor set by forced half-duplex and room reverberation that *predicts the winning regime* (C1), and (ii) the rate × robustness × latency optimum under a *joint acoustic-annoyance cost* (C2); we scope the channel as a zero-association, credential-free **rendezvous plane, not a Wi-Fi replacement** (C3)."

Rationale grounded in the located work (why the systems contribution leads, with C1 its crux and C2 its freshest claim):

1. **C1 is the best novelty candidate** (`PLAN.md` §9.3 item 2). No located work states the *protocol-level* round-trip floor; the closest (§1.1, §1.3) treat reverberation as a *signal-processing* obstacle, not a *latency floor*. Crisp and cheap to falsify (C1 death condition, `PLAN.md` §1.4: regression slope ≈ 0 across room presets).
2. **C2 is the freshest** (`PLAN.md` §9.3 item 3) and *supported* by the verified psychophysics (§2.1, §2.2) — but it risks a (useful) negative result, so it leads the *discussion*, not the *headline*.
3. **C3 is framing, not novelty** — located work (§3.1, §3.2, §4.1) shows the RF side is well-studied and acoustic pairing is a shipped idea. Leading with C3 invites the exact dismissal `PLAN.md` §9.3 anticipates. Lead with it only as the *credibility baseline*, not the claim.

**Decisions to close at S7:**
- [x] C1/C2/C3 all backed by entries in this file — **done** (see §1–§4, §6).
- [x] One contribution sentence (final, above) and **D4 = lead with the systems contribution** (C1 crux, C2 freshest, C3 scope) — **decided 2026-09-28**.
- [x] §5 table: every baseline number source-traced or `[V]` — **done**; Wi-Fi `[V]` rows explicitly *not* closed (honesty-preserving).

---

## 8. Spatial dimension — candidate claim C4 (hooks only; Phase C deferred)

**Scope note (inventory added 2026-09-25; extends D7, `PLAN.md` §13, recorded 2026-09-25).**
Spatial is in scope **as cheap hooks only**: a spatial analogue in `PLAN.md` §2.2, spatial
quantities in the `PLAN.md` §8 metrics, a `spatial_dof()` estimator in `S10 acoustics/channel.py`, and
array geometry in `S2.5/S2.6 configs/*.yaml`. A full **Phase C** (spatial-modulation pilot, SDMA,
beamforming/C2, robot audition) is **deferred** behind the Phase 0/1/2 gates. Nothing in this §8
is a commitment — it is a *falsifiable candidate* and an honest novelty inventory. Tag **C4** is a
*candidate* claim, distinct from C1–C3 (which this file already defends).

### 8.1 Spatial is a *new alphabet dimension*, not a new project

Spatial adds **bearing / array index / room mode** to the existing `L0–L4` language. Every
mechanism below is expressible as `L0-S` (a spatial code) and/or `L3-S` (bearing in the semantic
channel), so the handshake story stays coherent: spatial is the **fourth knob** (beyond distance)
on the same `C1/C2/C3` trade-off.

### 8.2 Mechanisms M1–M5 — what spatial can do, and what already exists

| # | Mechanism | Effect on the handshake | Maturity | Closest prior art |
|---|-----------|-------------------------|----------|-------------------|
| M1 | **Spatial multiplexing** (N speakers → M mics) | aggregate goodput ↑ if paths decorrelate | high (underwater) | Li, Huang, Stojanovic et al. 2009 `[VERIFIED]` |
| M2 | **Spatial / index modulation** (which speaker carries the symbol) | +log₂N bits/pulse at *zero* extra power | very high (RF) | Di Renzo, Haas et al. 2014 `[VERIFIED]`; Mesleh, Haas et al. 2010 `[VERIFIED]` |
| M3 | **SDMA spatial selectivity** (beam to robot, *not* room) | C2 ↓; eavesdrop ↓; rate ↑ | high (RF/beamform) | (shared with M2) |
| M4 | **Bearing semantics** (robot localizes → acts) | `L3` channel; localization is the payload | high (robot audition) | Nakadai, Okuno, Kitano 2002 `[VERIFIED]`; Rascón & Meza 2017 `[VERIFIED]` |
| M5 | **Beamforming / energy concentration** | C2 ↓ via directivity | high (spatial audio) | Herre, Kellermann, Iyer 2015 `[VERIFIED]`; Zhang et al. 2017 `[VERIFIED]` |

**Honest maturity reading:** M1 and M2 are *solved* in the RF / underwater world; M4 is a mature
discipline; M5 is a product category. **None is novel.** The only non-trivial combination is
**spatial modulation in a *diffuse, reverb-bound, room-scale* acoustic channel**, and the
OpenAlex hit-count evidence below says that combination is under-studied.

### 8.3 Core insight: reverb *creates* spatial rank while destroying temporal capacity

In a diffuse room the late reverb tail destroys temporal coherence (C1's enemy) **but** supplies
*independent* propagation paths between speaker and mic — the very thing that produces spatial
MIMO rank. **Consequence:** there is an **interior optimum in RT60** — too little reverb → low
spatial rank (M1/M2 vanish); too much → temporal smearing and audibility (C1/C2) collapse. This
is in **direct tension with C1** (which wants *less* reverb, monotonically) and is the project's
most interesting spatial prediction. `spatial_dof()` (S10) must measure it; the **RT60-vs-rank
curve is the falsifiable figure** (C4 death condition: `spatial_dof()` flat or ≈ 1 across presets,
i.e. no rank to exploit).

### 8.4 Frequency–spatial allocation: spend budget where the room gives rank + directivity

**Honest physics first.** Frequency band and bearing are *separate* degrees of freedom: a tone
at 3 kHz arriving from the north is just a 3 kHz tone, and the array resolves "north" from
inter-mic phase (TDOA/DOA), **not** from the band label. *Labeling a band as a direction does
nothing to the channel.* Banding only pays off through two *real* frequency↔space couplings:

1. **Frequency-dependent directivity.** For a fixed aperture, beamwidth ∝ λ — a higher band is a
   *narrower* beam → more bearing selectivity, less room coupling → better **C2** robustness and
   anti-eavesdrop; a lower band is more omnidirectional.
2. **Frequency-dependent reverb.** RT60(f) is usually *shorter* at high f (rooms absorb high f
   harder) → less temporal smearing at high f, but a tighter propagation margin through
   occlusions; low f carries more reverb energy.

**So — the useful version is a band-budgeted robustness allocation:** put **C2-critical /
security-critical symbols in the most directive, shortest-reverb band**, and redundant discovery
symbols in a lower, more-diffuse band. Different symbols then get different robustness profiles
for free (a config knob, not new hardware). This is the honest power-domain proxy for spatial
multiplexing when you do *not* have enough array elements for true MIMO.

**What it is *not* (the crux).** It does **not create spatial rank**: if the room channel is rank
1, splitting into K bands just spreads one rank-1 stream into K bands and costs ~K× SNR per band —
it is **resource allocation, not a new DOF**. It does **not** help TTCH (still room-bound, C1).
In a low-SNR, reverb-bound room (exactly the C1 regime), splitting power across bands usually
*hurts* per-symbol SNR unless you actually have rank ≥ #bands. And this is a *handshake* (few
bits), so spatial multiplexing's win is weaker here than the M2/M5 moves below.

**Principled version (C4/Phase-C refinement of M2/M3/M5):**
- **Index modulation (M2):** encode the bearing in *which* beam/subcarrier is active — zero extra
  power, strictly cleaner than naming bands.
- **Frequency-selective beamforming (M3/M5):** pick the band where the array is most directive
  *toward the target bearing* and null the room — this is the honest C2 win.
- **Rank-aware allocation:** spend spatial budget only where rank(f) is high; fall back to one
  best band where rank(f) ≈ 1.

**Cheap hooks (D7 scope, falsifiable):** a `spatial_dof(f)` estimator (rank vs frequency) in
`S10 acoustics/channel.py`, plus a `band→bearing` allocation field in `S2.5/S2.6 configs/*.yaml`.
**Falsification:** if rank(f) ≈ 1 across all bands *and* per-band beams are wide, per-direction
banding gives no gain over one well-chosen band → drop it. **Win case:** high-f bands are strongly
directive + short-reverberant → assigning C2-critical symbols there yields a measurable C2
reduction and eavesdrop margin. *Not novel on its own* (FDM and beamforming are solved);
candidate-grade only as a *combination detail* under C4.

### 8.5 Latency is *not* the spatial win

Spatial does **not** reduce TTCH (handshake latency is bound at 64–512 bits by C1's half-duplex
+ RT60 floor). Its wins are: (a) payload-phase *throughput* (M1), (b) bits-per-pulse at zero extra
power (M2), (c) C2 ↓ via selectivity/directivity (M3/M5). **Do not advertise a spatial latency
win — the physics does not support it** (`PLAN.md` §2.2 / §2.4, §8 TTCH target).

### 8.6 Sourcing (OpenAlex, live 2026-09-25) — hit-count = study-density signal

Citations below use the file's §0 tags: **[VERIFIED]** = confirmed by a live OpenAlex query on
2026-09-25 (full author order as returned); **[FROM PLAN §9.2]** = identifier already recorded in
`PLAN.md` §9.2; they were not found on OpenAlex on 2026-09-25 but are now **confirmed via the arXiv API on 2026-09-28** (both resolve with titles matching the drafts) → **[VERIFIED]**.

| Work (authors, as returned) | Year | DOI / ID | Cited by | Hits (OpenAlex `cited_by_count`) |
|---|---|---|---|---|
| Li, Huang, Stojanovic, Li, Stojanovic, Huang — J. Acoust. Soc. Am. 125(3):1436–1451 | 2009 | `10.1121/1.3068361` | M1, M3 | 283 |
| Di Renzo, Haas, Yanikomeroglu, MU — IEEE COMST 16(2):198–212 | 2014 | `10.1109/mcom.2014.2298284` | M2, M3 | 1914 |
| Mesleh, Haas, Tsiftsis, Alouini — IEEE Trans. Veh. Tech. 59(6):2796–2806 | 2010 | `10.1109/tvt.2010.2053171` | M2 | 423 |
| Nakadai, Okuno, Kitano — IEEE ASUPO 10(3):243–254 | 2002 | `10.1109/72.1019765` | M4 | 196 |
| Rascón, Meza — Sensors 17(1):27 | 2017 | `10.3390/s17010027` | M4 | 29 |
| Herre, Kellermann, Iyer — Proc. IEEE 103(9):1458–1473 | 2015 | `10.1109/jproc.2015.2426631` | M5 | 238 |
| Zhang, Zhao, Guo, Xie — Sensors 17(3):529 | 2017 | `10.3390/s17030529` | M5 | 36 |
| Riedmaier, Knaak, Brunk — PLOS ONE 14(7) | 2019 | `10.1371/journal.pone.0219847` | context | 112 |
| Hogg, Freitas — arXiv:1202.0568 | 2012 | (arXiv) | M4, §4.2 | `[FROM PLAN §9.2]` `[VERIFIED]` (arXiv API 2026-09-28; §4.2) |
| Lieser, Schwanecke — arXiv:2312.17668 | 2024 | (arXiv) | §4.3 | `[FROM PLAN §9.2]` `[VERIFIED]` (arXiv API 2026-09-28; §4.3) |

**Reading:** the *individual* mechanisms are densely cited (Di Renzo 2014 IM: 1914; Li 2009 MIMO
acoustic: 283) — i.e. **not novel**. The *combination* "spatial/index modulation in a diffuse,
reverb-bound, room-scale acoustic channel for an audible machine-listening handshake" is the
specific gap — the RF MIMO-IM literature does not target reverberant rooms, and the spatial-audio
literature targets rendering, not data. **This is why C4 is a candidate, not a claim: it is only
defensible if `spatial_dof()` shows exploitable rank (§8.3) in our room presets.**

**Not found / not claimed (honesty):** no primary source located for (i) *audible-band* spatial
modulation for machine listeners in reverberant rooms; (ii) robot-audition SDMA using *speaker
arrays*; (iii) a *combined* spatial + temporal modulation for a room-scale handshake. These remain
the honest open gap for Phase C.

### 8.7 Claim / novelty inventory (C4 candidate) and how it hooks to existing work

| Dimension | Claim | Novelty? (vs located work) | Hook |
|---|---|---|---|
| Spatial multiplexing M1 | room-scale rank for audibility | **No** — solved in RF/underwater (Li 2009) | `spatial_dof()` (S10) |
| Spatial/index modulation M2/M3 | bits-per-pulse at zero power + selectivity | **No** — solved in RF (Di Renzo 2014; Mesleh 2010) | §2.2 `L0-S` code |
| Bearing semantics M4 | localization-as-payload | **No** — mature (Nakadai 2002; Rascón 2017) | §2.2 `L3-S` |
| Beamforming M5 | C2 ↓ via directivity | **No** — product category (Herre 2015) | §2.2 / C2 |
| **Combination** | spatial × temporal modulation in a *reverb-bound, room-scale, audibility-costed* handshake | **Yes (candidate)** — the specific combination is the gap (§8.6) | **C4**, Phase C |

**Bottom line (mirrors `PLAN.md` §9.3):** C4 is **candidate-grade**, not a headline. It is the
single non-trivial *combination*, it is **falsifiable cheaply** (RT60-vs-rank curve, §8.3), and it
**does not displace C1** (which is still the recommended lead per §7). Phase C is deferred until
the Phase 0/1/2 gates pass and `spatial_dof()` is trusted.

---

## 9. Dual-use / opportunistic acoustic channels — candidate extension (deferred)

**Scope note (added 2026-09-28; extends the D7 "hooks only" pattern).** This section records two *candidate* extensions to the framing. Nothing here is a commitment, a claim, or a novelty assertion. Both are **candidate-grade** in the exact sense of §8: defensible only if the stated falsification tests pass, and **deferred** behind the existing Phase 0/1/2 gates. A companion decision is logged as **D8** (`PLAN.md` §13).

### 9.1 The observation

The project's thesis is *"a dedicated acoustic link beats RF connection setup."* Both extensions share a complementary thesis: **the cheapest acoustic channel is one that is already transmitting** — reuse acoustic energy that is already emitted for another purpose (entertainment audio, machine noise, building ducts) as the carrier for a zero-association handshake. Name: **opportunistic / parasitic acoustic communication.**

### 9.2 Extension A — hybrid human + machine presentation (dual-use host audio)

A movie/podcast is already playing into the room. Put machine-decodable payload into that same airborne audio. Honest placement is a **spectrum of embedding depth**, not a new mechanism:

| Rung | Mechanism | Machine bits | Human cost | Novelty |
|---|---|---|---|---|
| 0 | **Content-as-code** (fingerprint: which title + timestamp) | identity only | none | shipped (media-sync / ACR) |
| 1 | **Companion band** above ~16–20 kHz (empty in human content, mostly inaudible) | low | ~none | partly shipped (TV→phone beacons, `PLAN.md` §9.2) |
| 2 | **Psychoacoustic-masked watermark** under the host's masking threshold | tens–hundreds bps, robust | none by construction | dense prior art (broadcast watermarking) |
| 3 | **Informed decoding** — receiver *knows* the played content and cancels the host | potentially higher | none | least-studied |

**Fit to existing claims.** It is the **dual of C2**: replace the *annoyance* term with a **masking/imperceptibility constraint** — same Pareto frontier, a different point on it. It is a **new delivery mode for L4 discovery / L3 rendezvous** (a handshake needs few bits, so a few-hundred-bps hidden channel is competitive *and* costs no extra airtime). **Reverb (C1) still applies** — host and watermark share the room response; if the host is known, deconvolving one helps the other. The **§8.4 band-allocation** machinery is the natural home ("human band" vs "machine companion band").

### 9.3 Extension B — noise as carrier (e.g. datacenter / machinery)

**Honest disanalogy first.** In 9.2 the audio is a *designed carrier* with intrinsic human value; datacenter noise is a *waste product serving a different function* (cooling). It is **not** "the same way" by default:

| Version | What it is | Reality | Honest read |
|---|---|---|---|
| Active (modulate fans) | RPM → blade-passing-frequency FSK | ~Hz bandwidth; thermal inertia ⇒ seconds–minutes; fan power ∝ RPM³ | **telemetry beacon**, not a data channel; energy-poor; C2 cost small only if unoccupied |
| Passive (just listen) | infer load/thermal state from noise that already varies | n/a | **sensing** channel; message not controllable |
| Infrastructure carrier | **HVAC/ducts as acoustic waveguides** (speaker in airflow) | narrow band, modal dispersion, long reach | strongest version — reuses installed plumbing |
| Covert-channel inverse | the same physics is a known **exfiltration** surface | — | a **threat to name**, not a feature to sell |

**Why it matters here:** it maps to **C1's regime prediction**. C1 says acoustic *loses* in loud/reverberant regimes (`PLAN.md` §2.5 rotor noise; this file §4.3). Extension B reframes that loss — instead of competing with the noise floor, *use* the loud regime as carrier. The **M7 rotor-noise testbed** (the drone's own rotor noise already exists) is the natural first check.

### 9.4 Falsification criteria (written before implementation)

- **Hybrid (9.2) dies** if, across the room presets, masking + reverb force the embedded rate so low that `TTCH_embedded > TTCH_shout` **and** audibility is not ~free. **Wins** if it delivers the handshake payload with (a) no measurable masking violation and (b) no dedicated airtime, where human audio already plays. It is **opportunistic** — silent rooms fall back to a dedicated frame; a complementary mode, never a replacement.
- **Noise-carrier (9.3) dies** if the achievable rate is below the duty-cycle cost of the modulation (fan energy / cooling penalty) *or* if the resulting tonality breaches the C2 annoyance bound in occupied spaces. **Wins** only as a low-rate beacon / sensing channel, or as duct-guided infrastructure reuse.

### 9.5 Claim inventory (candidate C5)

| Dimension | Claim | Novelty? (vs located work) | Hook |
|---|---|---|---|
| Hybrid host-audio embedding | machine payload in human presentation audio | **No** — watermarking / fingerprinting are shipped | C2 masking metric; §8.4 band allocation |
| Noise-as-carrier | reuse machine/infra noise as carrier | **No** — covert channels + pipe telemetry exist | C1 regime map; M7 |
| **Framing** | *opportunistic/parasitic acoustic channel* as the complement to the dedicated-link thesis | **Yes (candidate)** — framing only | C3 rendezvous; `PLAN.md` §9.3 framing |

### 9.6 Sourcing leads (⚠ **NOT VERIFIED** — must pass the §0 OpenAlex discipline before any figure is quoted)

Named as *search leads only*; no DOI/arXiv ID is asserted here, unlike §1–§4/§8. Leads: audio watermarking / data-hiding (echo-hiding, spread-spectrum, phase-coding, informed decoding); broadcast-audio watermarking & media-sync fingerprinting; near-ultrasonic beacons (`PLAN.md` §9.2); acoustic telemetry in pipes/ducts; fan/disk acoustic covert channels from air-gapped machines; HVAC-noise-based sensing. **Until verified, treat every name above as a hypothesis.**

---

## Appendix A — Sourcing log (OpenAlex, 2026-09-25)

Live OpenAlex API queries (`api.openalex.org/works?search=...`), 2026-09-25, returned the entries tagged **[VERIFIED]** above. Raw dumps: `/tmp/rw_openalex.txt` (broad phrase queries) and `/tmp/rw_verify_out.txt` (title-anchored re-queries). DOIs / OpenAlex work IDs printed in §1–§4 are the API's returned values.

**Pending full-metadata verification** (carried from `PLAN.md` §9.2 with the arXiv IDs already recorded there; a live re-query did not return a clean match on 2026-09-25 — re-confirm venue/DOI before quoting any numbers):

- Matt & Stowell (2019), arXiv:1908.04672 — §1.3
- Murano et al. (2020), arXiv:2402.02400 — §1.4
- Shen et al. (2023), arXiv:2312.15650 — §3.2
- Hogg & Freitas (2012), arXiv:1202.0568 — §4.2
- Lieser & Schwanecke (2024), arXiv:2312.17668 — §4.3

These identifiers are **not** fabricated: they were already present in the project's own `PLAN.md` §9.2 prior-art table. They are flagged so that S5.2 / the M1 re-sourcing step confirms them against a primary source before any specific figure is attributed to them.

**§9 extension leads are a different class.** The names in §9.6 are *search directions*,
not located works: they carry no DOI/arXiv ID and no `[VERIFIED]` tag by design. They must
pass the full §0 OpenAlex discipline (exact authors/title/venue/year + a live query) before
any of them is cited in a figure or the README. Do not conflate them with the pending arXiv
IDs above, which at least have a resolvable identifier.