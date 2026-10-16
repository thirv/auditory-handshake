"""Multi-device slotted-ALOHA contention model and its TTCH inflation.

When more than one device must rendezvous through the acoustic channel at the
same time, the single-link TTCH model (``metrics.ttch.ttch_rtt_s``) is no longer
enough: a transmitting device must first win a *clean* slot against the other
``N-1`` devices before its preamble+frame can land. This module models that
contention with the standard **slotted ALOHA** clean-slot probability and
reports how it inflates the TTCH *while preserving the C1 RT60-sensitivity*
(PLAN.md §2.4 / §1.4, M6). It **reuses** the existing TTCH and
``regress_rt60_sensitivity`` building blocks rather than re-deriving the
round-trip formula, so it stays in lockstep with the single-link C1 model.

Model
-----
A persistent requester contending with ``N-1`` other devices, each of which
transmits with probability ``p_tx`` per slot. Its slot is *clean* (it wins) iff
none of the other ``N-1`` transmits: ``P(clean) = (1 - p_tx) ** (N - 1)``.

The attempts until the first clean slot are Geometric with mean
``1 / P(clean) = (1 / (1 - p_tx)) ** (N - 1)`` (capped at ``n_slots_max``). A
``base_turnarounds``-way handshake (PLAN.md §2.4 gives 2) needs that many
successful accesses — one per direction — so the expected total turnarounds are
``T = base_turnarounds · expected_slots`` and the contention-inflated TTCH is the
existing PLAN.md §2.4 model evaluated at ``turnarounds = T``:

    TTCH = T · (T_preamble + T_frame + k·RT60) + T_proc

For ``N == 1`` this reduces exactly to the single-link baseline
(``T == base_turnarounds == 2``), a strict super-case of ``sims/06``. Because
``k·RT60`` sits *inside* each contended turnaround, contention **amplifies** the
RT60 sensitivity — ``d(TTCH)/d(RT60) = T · k`` grows with ``N`` while staying
positive — so the C1 crux survives and *strengthens* under contention.

All functions are pure (no I/O, no randomness) and mypy-clean;
``sims/07_n_device_contention.py`` builds the N × RT60 tables and figures.
"""

from __future__ import annotations

from auditory_handshake.metrics.ttch import (
    DEFAULT_K,
    DEFAULT_T_PROC_S,
    DEFAULT_TURNAROUNDS,
    ttch_rtt_s,
)

__all__ = [
    "T_PREAMBLE_S",
    "T_FRAME_S",
    "DEFAULT_P_TX",
    "DEFAULT_N_SLOTS_MAX",
    "clean_slot_probability",
    "expected_slots",
    "expected_turnarounds",
    "ttch_s",
]

# PLAN.md §2.4 per-turnaround timing constants (nominal), mirroring the
# checkpoint values in ``metrics/ttch.py`` (1-way preamble 1.25 ms; 32-bit frame
# @ 1024 bit/s = 31.25 ms). Exposed so sims/tests can override for sweeps.
T_PREAMBLE_S = 0.00125  # 1-way preamble detection latency (s)
T_FRAME_S = 32.0 / 1024.0  # 32-bit control frame @ 1024 bit/s = 31.25 ms (s)

# Contention defaults (overridable per call).
DEFAULT_P_TX = 0.20  # per-slot transmit probability of a contending device
DEFAULT_N_SLOTS_MAX = 20.0  # cap on clean-slot attempts before a device times out


def clean_slot_probability(
    n_devices: int,
    p_tx: float = DEFAULT_P_TX,
) -> float:
    """Probability a given device's slot is clean (it wins) under slotted ALOHA.

    ``(1 - p_tx) ** (n_devices - 1)``: the device wins iff none of the other
    ``n_devices - 1`` contending devices transmits in the same slot.

    Parameters
    ----------
    n_devices:
        Total number of contending devices ``N >= 1`` (including the requester).
    p_tx:
        Per-slot transmit probability of each of the other devices, in ``(0, 1)``.

    Returns
    -------
    float
        Clean-slot probability in ``(0, 1]``; exactly ``1.0`` for ``n_devices == 1``.
    """
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if not (0.0 < p_tx < 1.0):
        raise ValueError(f"p_tx must be in (0, 1), got {p_tx}")
    return float(1.0 - p_tx) ** (n_devices - 1)


def expected_slots(
    n_devices: int,
    p_tx: float = DEFAULT_P_TX,
    n_slots_max: float = DEFAULT_N_SLOTS_MAX,
) -> float:
    """Expected clean-slot attempts for one channel access (Geometric mean).

    ``1 / clean_slot_probability(n_devices, p_tx) =
    (1 / (1 - p_tx)) ** (n_devices - 1)``, capped at ``n_slots_max``. For
    ``n_devices == 1`` this is exactly ``1`` (no contention, a single clean slot).

    Parameters
    ----------
    n_devices:
        Total contending devices ``N >= 1``.
    p_tx:
        Per-slot transmit probability of each other device, in ``(0, 1)``.
    n_slots_max:
        Cap on the expected clean-slot attempts (a timeout bound), ``>= 1``.
    """
    if n_slots_max < 1.0:
        raise ValueError(f"n_slots_max must be >= 1, got {n_slots_max}")
    p_clean = clean_slot_probability(n_devices, p_tx)
    return min(1.0 / p_clean, n_slots_max)


def expected_turnarounds(
    n_devices: int,
    base_turnarounds: float = DEFAULT_TURNAROUNDS,
    p_tx: float = DEFAULT_P_TX,
    n_slots_max: float = DEFAULT_N_SLOTS_MAX,
) -> float:
    """Expected total transmission turnarounds to complete the handshake.

    ``base_turnarounds · expected_slots(n_devices, p_tx, n_slots_max)`` — the
    handshake has ``base_turnarounds`` directions (PLAN.md §2.4 gives 2) and each
    must win a clean slot under contention. For ``n_devices == 1`` this equals
    ``base_turnarounds`` (the uncontended baseline).

    Parameters
    ----------
    n_devices:
        Total contending devices ``N >= 1``.
    base_turnarounds:
        Uncontended number of transmit turnarounds (``>= 1``); 2 = 2-way.
    p_tx:
        Per-slot transmit probability of each other device, in ``(0, 1)``.
    n_slots_max:
        Cap on the expected clean-slot attempts per access, ``>= 1``.
    """
    if base_turnarounds < 1.0:
        raise ValueError(f"base_turnarounds must be >= 1, got {base_turnarounds}")
    return float(base_turnarounds) * expected_slots(n_devices, p_tx, n_slots_max)


def ttch_s(
    n_devices: int,
    rt60_s: float,
    t_preamble_s: float = T_PREAMBLE_S,
    t_frame_s: float = T_FRAME_S,
    k: float = DEFAULT_K,
    t_proc_s: float = DEFAULT_T_PROC_S,
    base_turnarounds: float = DEFAULT_TURNAROUNDS,
    p_tx: float = DEFAULT_P_TX,
    n_slots_max: float = DEFAULT_N_SLOTS_MAX,
) -> float:
    """Contention-inflated TTCH for ``n_devices`` devices, in seconds.

    Evaluates the existing PLAN.md §2.4 model (
    :func:`auditory_handshake.metrics.ttch.ttch_rtt_s`) at
    ``turnarounds = expected_turnarounds(...)`` so it stays in lockstep with the
    C1 baseline: ``TTCH = T · (T_preamble + T_frame + k·RT60) + T_proc`` with
    ``T = base_turnarounds · (1/(1-p_tx))**(N-1)`` (capped). For ``n_devices == 1``
    this is exactly the single-link baseline (``T == base_turnarounds``).

    Parameters
    ----------
    n_devices:
        Total contending devices ``N >= 1``.
    rt60_s:
        Room reverberation time in seconds (``>= 0``).
    t_preamble_s, t_frame_s, k, t_proc_s:
        Per-turnaround timing constants (see :mod:`auditory_handshake.metrics.ttch`).
    base_turnarounds:
        Uncontended number of transmit turnarounds (``>= 1``); 2 = 2-way.
    p_tx:
        Per-slot transmit probability of each other device, in ``(0, 1)``.
    n_slots_max:
        Cap on the expected clean-slot attempts per access, ``>= 1``.

    Returns
    -------
    float
        Contention-inflated ``TTCH`` in seconds.
    """
    turnarounds = expected_turnarounds(n_devices, base_turnarounds, p_tx, n_slots_max)
    return ttch_rtt_s(
        rt60_s,
        t_preamble_s,
        t_frame_s,
        t_proc_s=t_proc_s,
        k=k,
        turnarounds=turnarounds,
    )
