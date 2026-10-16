"""Two-way acoustic handshake round-trip timing (PLAN.md §2.4, §1.4, M6).

Claim C1 (PLAN.md §1.2) is that the acoustic handshake **must work in the
noisiest, most reverberant room**, so its turnaround time (TTCH) must *scale*
with the room's reverberation time (RT60). PLAN.md §1.4 makes this a falsifiable
criterion:

    "C1 dies if RTT is statistically insensitive — or worse, anti-correlated —
     to RT60."

The M4 one-way preamble proxy cannot answer that, because a *one-way* detection
latency never contains the **half-duplex turnaround guard**: after a node
transmits, the room keeps ringing and the medium cannot be reused until the
reverb decay falls below a threshold. That guard is the reverb-dependent term
and it only exists in a genuine **two-leg** exchange. This module supplies the
pure (I/O-free) building blocks to assemble such a round trip from *measured*
per-leg stage times and to apply the canonical §1.4 falsification test to it:

- :class:`TwoWayLeg` — the measured stage times of one transmit→detect leg.
- :func:`rtt_s` — the total two-way round-trip time from two legs.
- :func:`regress_ttch_vs_rt60` — a thin delegate to the canonical
  :func:`~auditory_handshake.metrics.ttch.regress_rt60_sensitivity`.

Two anti-tautology rules that make this a *measurement* rather than a model
identity (the sim_06/sim_07 defect):

1. A leg's ``t_guard_s`` must come from the RIR's **measured** Schroeder
   energy-decay curve (``acoustics.channel.echo_tail_guard_s``), **not** an
   assumed ``k·RT60`` constant.
2. :func:`regress_ttch_vs_rt60` must be called with the room's **measured**
   Schroeder RT60 (``metrics.channel_metrics.rt60_schroeder``), **not** the
   target RT60 fed into the RIR synthesiser.

All times are in seconds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .ttch import RT60Sensitivity, regress_rt60_sensitivity

__all__ = [
    "TwoWayLeg",
    "rtt_s",
    "regress_ttch_vs_rt60",
]


@dataclass(frozen=True)
class TwoWayLeg:
    """One measured transmit→detect leg of a half-duplex round trip.

    Every time field is in seconds and is a *measured* quantity from a
    simulation or device, not a design assumption.

    Fields
    ------
    t_tx_s:
        On-air duration of the leg's preamble + control frame.
    t_guard_s:
        Half-duplex turnaround guard for this leg: the time the medium must be
        left free for the reverb decay to fall below threshold. This is the
        C1 reverb-dependent term and **must** be measured from the RIR's
        Schroeder energy-decay curve (see
        :func:`auditory_handshake.acoustics.channel.echo_tail_guard_s`), not an
        assumed ``k·RT60``.
    t_detect_s:
        Preamble matched-filter decision latency for this leg.
    t_proc_s:
        Fixed decode / DSP / codec time for this leg.
    decoded:
        Whether the receiver both detected the preamble **and** the frame passed
        its CRC check. A leg that failed still contributes its
        transmit+guard+detect time; a retry is modelled by the caller adding the
        failed attempt's times to the next.
    """

    t_tx_s: float
    t_guard_s: float
    t_detect_s: float
    t_proc_s: float
    decoded: bool


def _check_non_negative(leg: TwoWayLeg, label: str) -> None:
    """Raise ``ValueError`` if any time field is negative or non-finite."""
    for name in ("t_tx_s", "t_guard_s", "t_detect_s", "t_proc_s"):
        v = getattr(leg, name)
        if not (v >= 0.0):
            raise ValueError(f"{label}.{name} must be >= 0 and finite, got {v!r}")


def rtt_s(forward: TwoWayLeg, reverse: TwoWayLeg) -> float:
    """Total two-way handshake round-trip time, in seconds (PLAN.md §2.4).

    ``RTT = Σ(forward leg stages) + Σ(reverse leg stages)`` where each leg's
    stage sum is ``t_tx_s + t_guard_s + t_detect_s + t_proc_s``.

    The ``t_guard`` terms are the reverb-dependent half-duplex turnaround that a
    one-way M4 proxy cannot contain, so this is the quantity the PLAN.md §1.4
    falsification test is applied to (see :func:`regress_ttch_vs_rt60`).

    A leg whose ``decoded`` flag is ``False`` still contributes its full
    stage-sum: a failed detection or CRC still consumed transmit + guard +
    detect time. Model a retransmission by passing the *cumulative* failed
    attempt times in the leg that ultimately succeeded.

    Parameters
    ----------
    forward:
        The A→B leg.
    reverse:
        The B→A (ACK) leg.

    Returns
    -------
    float
        Total round-trip time in seconds.

    Raises
    ------
    ValueError
        If any stage time is negative or non-finite.
    """
    _check_non_negative(forward, "forward")
    _check_non_negative(reverse, "reverse")
    return (
        forward.t_tx_s
        + forward.t_guard_s
        + forward.t_detect_s
        + forward.t_proc_s
        + reverse.t_tx_s
        + reverse.t_guard_s
        + reverse.t_detect_s
        + reverse.t_proc_s
    )


def regress_ttch_vs_rt60(
    rt60_s: Sequence[float],
    ttch_s: Sequence[float],
) -> RT60Sensitivity:
    """PLAN.md §1.4 falsification test applied to a *measured* two-way handshake.

    Thin delegate to
    :func:`auditory_handshake.metrics.ttch.regress_rt60_sensitivity` so that C1
    uses the **same** canonical falsification code as the rest of the repo (an
    OLS fit of measured latency on RT60, with Pearson ``r``, a ``t``-statistic
    and a plain-English verdict).

    To keep this an *empirical* test rather than the sim_06/sim_07 tautology,
    callers must pass:

    - ``rt60_s``: each room's **measured** Schroeder RT60
      (``metrics.channel_metrics.rt60_schroeder``), **not** the target RT60 fed
      into the RIR synthesiser;
    - ``ttch_s``: the measured per-room round-trip turnaround
      (e.g. the per-room ``rtt_s`` statistic from the simulation harness).

    A flat (``slope ≈ 0``) or negative result is a valid negative: per PLAN.md
    §1.4 that **kills** C1 and must be reported as such.

    Parameters
    ----------
    rt60_s:
        Per-room measured reverberation times in seconds (1-D, non-zero variance).
    ttch_s:
        Per-room measured two-way round-trip turnarounds in seconds, aligned
        with ``rt60_s``.

    Returns
    -------
    RT60Sensitivity
        The canonical verdict object (``verdict`` ∈ {CORROBORATED,
        WEAKLY_CORROBORATED, AT_RISK, CONTRADICTED, INCONCLUSIVE}).
    """
    return regress_rt60_sensitivity(rt60_s, ttch_s)
