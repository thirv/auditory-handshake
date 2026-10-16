# Auditory Handshake — Protocol Specification (SKELETON)

> **Status: skeleton (S8.1).** This is the *structural outline* of the protocol
> language — headings and intent only, **no normative content yet**. Each layer
> is filled in only when its milestone produces the measurement that defines it.
> See `PLAN.md` §3 for the layer model and `PLAN.md` §7 for what is actually
> implemented today (currently: RF baselines + capacity ceiling only; no acoustic
> signal, waveform, or handshake state machine yet).
>
> This file is deliberately a placeholder. The project does not ship "a protocol
> spec" — it ships the *measurement* that would justify one (C1/C2), and this
> document is the frame that measurement will fill.

## L0 — Channel access
*Intent:* how a device seizes the acoustic medium — preamble format, duty cycle,
contention / backoff, half-duplex turnaround.
*(content: pending — M3/M4/M6)*

## L1 — Data (MCS)
*Intent:* modulation and coding set: constellation, FEC, rate ladder, and the
rate↔robustness trade.
*(content: pending — M5)*

## L2 — Link
*Intent:* ACK/ARQ, channel-state feedback, and the PHY-constraint handshake that
lets both sides agree on what the room actually allows.
*(content: pending — M6/M7)*

## L3 — Handshake
*Intent:* the connection-establishment state machine and the bearing (position)
semantics — what the handshake *is* (latency floor), not a new physical channel.
*(content: pending — M6/M9)*

## L4 — Security
*Intent:* anti-impersonation, session-key establishment, and anti-overshadowing.
Explicitly **not** the novelty (`RELATED_WORK.md` §4: Putz 2020 already ships
acoustic pairing + security).
*(content: pending — M8)*
