"""Handshake control-plane: protocol FSM and handshake frame format (Phase C / M6).

This module is the *state machine* part of the acoustic handshake: a small,
pure-Python finite-state machine that a physical endpoint (or a simulation)
drives with events, plus the fixed-length *handshake frame* (identity,
negotiation nonce, proposed MCS index) protected by CRC-16 (CCITT).

Design constraints (PLAN §4 working rules)

* **Deterministic** — no global RNG, no I/O, no wall-clock.  The caller
  supplies the event *time* (seconds, any monotonic domain); the FSM only
  compares timestamps.  This keeps the control plane fully unit-testable.
* **Pure** — :func:`step` never mutates its inputs and returns a fresh
  :class:`HandshakeState`; the FSM table itself is a module-level constant
  so transitions can be inspected and asserted against in tests.
* **Audio-free** — the FSM knows nothing about waveforms.  The acoustic
  side (preamble detection, MFSK modem, channel model in
  :mod:`auditory_handshake.phy`) produces the *events* (:data:`PREAMBLE_DETECTED`,
  :data:`FRAME_OK`, :data:`TIMEOUT`, ...) and executes the *actions*
  (``request_tx`` / ``retry`` / ``abort``) that come back.

Frame layout (12 bytes, little-endian):

========================  ====  =========================================
offset                    size  field
========================  ====  =========================================
0                         8     identity (device name, UTF-8, exactly 8 B)
8                         1     nonce (handshake round / replay token)
9                         1     proposed MCS index (0-3, 4-FSK levels)
10                        2     CRC-16/CCITT over bytes 0..9
========================  ====  =========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

HANDSHAKE_FRAME_LEN = 12
_IDENTITY_LEN = 8
_MAX_MCS = 3

_CRC16_POLY = 0x1021
_CRC16_INIT = 0xFFFF


class HandshakeStateName(Enum):
    """FSM states of the acoustic handshake control plane."""

    IDLE = "idle"
    LISTENING = "listening"
    HANDSHAKE = "handshake"
    READY = "ready"
    DATA = "data"
    FAILED = "failed"


class HandshakeEvent(Enum):
    """Events that drive the handshake FSM (time supplied by the caller)."""

    ARM = "arm"
    PREAMBLE_DETECTED = "preamble_detected"
    FRAME_OK = "frame_ok"
    FRAME_CORRUPT = "frame_corrupt"
    ACK_SENT = "ack_sent"
    DATA_SENT = "data_sent"
    DATA_FINISHED = "data_finished"
    TIMEOUT = "timeout"
    RESET = "reset"


class HandshakeError(ValueError):
    """Raised when an event is not valid in the current state."""


@dataclass(frozen=True)
class HandshakeFrame:
    """A decoded, CRC-validated handshake frame.

    Attributes
    ----------
    identity:
        The 8-byte device identity (``bytes``) of the far endpoint.
    nonce:
        Handshake round / replay token (``0..255``).
    mcs:
        Proposed MCS index (``0..3`` for 4-FSK modulation levels).
    """

    identity: bytes
    nonce: int
    mcs: int

    def to_bytes(self) -> bytes:
        """Serialise back into the 12-byte wire layout (CRC recomputed)."""
        return encode_handshake_frame(self.identity, self.nonce, self.mcs)

    def to_dict(self) -> dict[str, object]:
        """Round-trip-friendly dict (identity as UTF-8 text when possible)."""
        try:
            identity = self.identity.decode("utf-8")
        except UnicodeDecodeError:
            identity = self.identity.hex()
        return {"identity": identity, "nonce": self.nonce, "mcs": self.mcs}


def crc16_ccitt(data: bytes | bytearray | memoryview) -> int:
    """Compute CRC-16/CCITT (poly ``0x1021``, init ``0xFFFF``, no reflect).

    Deterministic and dependency-free (``crc16_ccitt(b'123456789') == 0x29B1``).
    """
    crc = _CRC16_INIT
    for byte in bytes(data):
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_handshake_frame(identity: bytes | str, nonce: int, mcs: int) -> bytes:
    """Encode a 12-byte handshake frame (identity + nonce + mcs + CRC-16).

    Raises
    ------
    ValueError
        If ``identity`` is not exactly 8 bytes of UTF-8, ``nonce`` is not in
        ``0..255``, or ``mcs`` is not in ``0..3``.
    """
    if isinstance(identity, str):
        identity = identity.encode("utf-8")
    ident = bytes(identity)
    if len(ident) != _IDENTITY_LEN:
        raise ValueError(
            f"identity must be exactly {_IDENTITY_LEN} bytes of UTF-8, got {len(ident)}"
        )
    if not isinstance(nonce, int) or isinstance(nonce, bool) or not 0 <= nonce <= 255:
        raise ValueError(f"nonce must be an int in 0..255, got {nonce!r}")
    if not isinstance(mcs, int) or isinstance(mcs, bool) or not 0 <= mcs <= _MAX_MCS:
        raise ValueError(f"mcs must be an int in 0..{_MAX_MCS}, got {mcs!r}")

    body = ident + bytes((nonce, mcs))
    return body + crc16_ccitt(body).to_bytes(2, "big")


def decode_handshake_frame(frame: bytes) -> HandshakeFrame:
    """Decode and CRC-verify a 12-byte handshake frame.

    Raises
    ------
    ValueError
        If the frame is not exactly 12 bytes or its CRC does not match.
    """
    raw = bytes(frame)
    if len(raw) != HANDSHAKE_FRAME_LEN:
        raise ValueError(f"handshake frame must be {HANDSHAKE_FRAME_LEN} bytes, got {len(raw)}")
    body, crc_bytes = raw[:10], raw[10:]
    expected = int.from_bytes(crc_bytes, "big")
    actual = crc16_ccitt(body)
    if expected != actual:
        raise ValueError(
            f"handshake frame CRC mismatch: expected 0x{expected:04X}, got 0x{actual:04X}"
        )
    return HandshakeFrame(identity=raw[:_IDENTITY_LEN], nonce=raw[8], mcs=raw[9])


# --------------------------------------------------------------------------
# FSM.
# --------------------------------------------------------------------------

#: ``(state, event) -> (next_state, action)``; action is one of
#: ``"request_tx"`` (send a handshake/data frame), ``"retry"`` (re-arm the
#: listen window), ``"abort"`` (give up, report FAILED) or ``"none"``.
TRANSITIONS: dict[tuple[HandshakeStateName, HandshakeEvent], tuple[HandshakeStateName, str]] = {
    (HandshakeStateName.IDLE, HandshakeEvent.ARM): (
        HandshakeStateName.LISTENING,
        "none",
    ),
    (HandshakeStateName.IDLE, HandshakeEvent.RESET): (
        HandshakeStateName.IDLE,
        "none",
    ),
    (HandshakeStateName.LISTENING, HandshakeEvent.PREAMBLE_DETECTED): (
        HandshakeStateName.HANDSHAKE,
        "request_tx",
    ),
    (HandshakeStateName.LISTENING, HandshakeEvent.TIMEOUT): (
        HandshakeStateName.IDLE,
        "retry",
    ),
    (HandshakeStateName.LISTENING, HandshakeEvent.RESET): (
        HandshakeStateName.IDLE,
        "none",
    ),
    (HandshakeStateName.HANDSHAKE, HandshakeEvent.FRAME_OK): (
        HandshakeStateName.READY,
        "request_tx",
    ),
    (HandshakeStateName.HANDSHAKE, HandshakeEvent.FRAME_CORRUPT): (
        HandshakeStateName.LISTENING,
        "retry",
    ),
    (HandshakeStateName.HANDSHAKE, HandshakeEvent.TIMEOUT): (
        HandshakeStateName.LISTENING,
        "retry",
    ),
    (HandshakeStateName.HANDSHAKE, HandshakeEvent.RESET): (
        HandshakeStateName.IDLE,
        "none",
    ),
    (HandshakeStateName.READY, HandshakeEvent.ACK_SENT): (
        HandshakeStateName.DATA,
        "request_tx",
    ),
    (HandshakeStateName.READY, HandshakeEvent.TIMEOUT): (
        HandshakeStateName.IDLE,
        "retry",
    ),
    (HandshakeStateName.READY, HandshakeEvent.RESET): (
        HandshakeStateName.IDLE,
        "none",
    ),
    (HandshakeStateName.DATA, HandshakeEvent.DATA_SENT): (
        HandshakeStateName.DATA,
        "request_tx",
    ),
    (HandshakeStateName.DATA, HandshakeEvent.DATA_FINISHED): (
        HandshakeStateName.IDLE,
        "none",
    ),
    (HandshakeStateName.DATA, HandshakeEvent.TIMEOUT): (
        HandshakeStateName.IDLE,
        "retry",
    ),
    (HandshakeStateName.DATA, HandshakeEvent.RESET): (
        HandshakeStateName.IDLE,
        "none",
    ),
}


@dataclass(frozen=True)
class HandshakeConfig:
    """Tunables for the handshake FSM (all durations in seconds).

    ``listen_timeout_s`` bounds each listen window before a retry;
    ``max_retries`` is the retry budget after which a TIMEOUT takes the FSM
    to :attr:`HandshakeStateName.FAILED` instead of back to IDLE/LISTENING.
    """

    listen_timeout_s: float = 0.5
    handshake_timeout_s: float = 1.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if self.listen_timeout_s <= 0:
            raise ValueError(f"listen_timeout_s must be > 0, got {self.listen_timeout_s}")
        if self.handshake_timeout_s <= 0:
            raise ValueError(f"handshake_timeout_s must be > 0, got {self.handshake_timeout_s}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass(frozen=True)
class HandshakeState:
    """Immutable snapshot of the FSM at some instant.

    Attributes
    ----------
    state:
        Current state name.
    retries:
        Number of retries already consumed (reset on a successful handshake
        round-trip and on RESET from FAILED).
    last_event_at_s:
        Timestamp (caller's domain, seconds) of the last event applied.
    """

    state: HandshakeStateName = HandshakeStateName.IDLE
    retries: int = 0
    last_event_at_s: float = 0.0


@dataclass(frozen=True)
class HandshakeStep:
    """Result of one :func:`step`: the new state plus the action to execute.

    ``action`` is one of ``"none"``, ``"request_tx"``, ``"retry"``,
    ``"abort"`` and is for the acoustic/transport layer to act on; the FSM
    itself performs no I/O.
    """

    state: HandshakeState
    action: str


def initial_state(config: HandshakeConfig | None = None) -> HandshakeState:
    """Return a fresh IDLE state (``config`` is accepted for API symmetry)."""
    if config is not None:
        # Re-validate so misconfiguration surfaces at construction time.
        HandshakeConfig(
            listen_timeout_s=config.listen_timeout_s,
            handshake_timeout_s=config.handshake_timeout_s,
            max_retries=config.max_retries,
        )
    return HandshakeState(state=HandshakeStateName.IDLE, retries=0, last_event_at_s=0.0)


def step(
    current: HandshakeState,
    event: HandshakeEvent,
    at_s: float,
    *,
    config: HandshakeConfig | None = None,
) -> HandshakeStep:
    """Apply ``event`` (at time ``at_s``) to ``current``; return new state + action.

    Deterministic and pure: identical inputs always give identical outputs.
    On a TIMEOUT the retry budget (``config.max_retries``) is consumed; once
    exhausted the FSM lands in :attr:`HandshakeStateName.FAILED` with action
    ``"abort"``.  Any event not in :data:`TRANSITIONS` for the current state
    raises :class:`HandshakeError`.
    """
    if at_s < current.last_event_at_s:
        raise HandshakeError(f"event at {at_s}s is before last event at {current.last_event_at_s}s")
    cfg = config if config is not None else HandshakeConfig()
    if event is HandshakeEvent.RESET and current.state is HandshakeStateName.FAILED:
        # Escape hatch: a FAILED handshake can always be cleared back to IDLE.
        return HandshakeStep(
            state=HandshakeState(
                state=HandshakeStateName.IDLE,
                retries=0,
                last_event_at_s=at_s,
            ),
            action="none",
        )
    try:
        next_state, action = TRANSITIONS[(current.state, event)]
    except KeyError as exc:
        raise HandshakeError(f"event {event.name} not valid in state {current.state.name}") from exc

    retries = current.retries
    if event is HandshakeEvent.TIMEOUT:
        if current.retries + 1 > cfg.max_retries:
            return HandshakeStep(
                state=HandshakeState(
                    state=HandshakeStateName.FAILED,
                    retries=current.retries,
                    last_event_at_s=at_s,
                ),
                action="abort",
            )
        retries = current.retries + 1
    elif event in (HandshakeEvent.FRAME_OK, HandshakeEvent.DATA_FINISHED):
        # A completed round-trip or a clean finish resets the retry budget.
        retries = 0

    return HandshakeStep(
        state=HandshakeState(state=next_state, retries=retries, last_event_at_s=at_s),
        action=action,
    )


__all__ = [
    "HANDSHAKE_FRAME_LEN",
    "HandshakeConfig",
    "HandshakeError",
    "HandshakeEvent",
    "HandshakeFrame",
    "HandshakeState",
    "HandshakeStateName",
    "HandshakeStep",
    "TRANSITIONS",
    "crc16_ccitt",
    "decode_handshake_frame",
    "encode_handshake_frame",
    "initial_state",
    "step",
]
