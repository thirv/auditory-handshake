"""Unit tests for the handshake FSM and frame codec (:mod:`auditory_handshake.protocol`).

Runs as plain pytest: ``pytest tests/test_handshake_fsm.py``.
Covers the full state graph, the retry/timeout/failure paths, and the
CRC-protected 12-byte handshake frame (round-trip + corruption rejection).
"""

from __future__ import annotations

import pytest

from auditory_handshake.protocol import (
    HANDSHAKE_FRAME_LEN,
    TRANSITIONS,
    HandshakeConfig,
    HandshakeError,
    HandshakeEvent,
    HandshakeStateName,
    crc16_ccitt,
    decode_handshake_frame,
    encode_handshake_frame,
    initial_state,
    step,
)


def drive(*events: tuple[HandshakeEvent, float], **cfg) -> object:
    """Walk the FSM through ``(event, at_s)`` pairs and return the final step."""
    config = HandshakeConfig(**cfg)
    state = initial_state()
    last = None
    for event, at in events:
        out = step(state, event, at, config=config)
        state = out.state
        last = out
    return last


def test_crc_known_vector() -> None:
    assert crc16_ccitt(b"123456789") == 0x29B1
    assert crc16_ccitt(b"") == 0xFFFF


def test_frame_roundtrip() -> None:
    raw = encode_handshake_frame("dev-0001", nonce=7, mcs=2)
    assert len(raw) == HANDSHAKE_FRAME_LEN == 12
    frame = decode_handshake_frame(raw)
    assert frame.identity == b"dev-0001"
    assert frame.nonce == 7
    assert frame.mcs == 2
    # Encode/decode is stable and to_bytes reproduces the wire bytes.
    assert frame.to_bytes() == raw
    assert frame.to_dict() == {"identity": "dev-0001", "nonce": 7, "mcs": 2}


def test_frame_rejects_bad_length() -> None:
    with pytest.raises(ValueError, match="12 bytes"):
        decode_handshake_frame(b"\x00" * 11)
    with pytest.raises(ValueError, match="12 bytes"):
        decode_handshake_frame(b"\x00" * 13)


def test_frame_rejects_corrupt_crc() -> None:
    raw = bytearray(encode_handshake_frame("dev-0001", 0, 0))
    raw[5] ^= 0x01  # flip one identity bit
    with pytest.raises(ValueError, match="CRC mismatch"):
        decode_handshake_frame(bytes(raw))
    raw = bytearray(encode_handshake_frame("dev-0001", 0, 0))
    raw[-1] ^= 0x80  # flip a CRC bit
    with pytest.raises(ValueError, match="CRC mismatch"):
        decode_handshake_frame(bytes(raw))


def test_frame_validation() -> None:
    with pytest.raises(ValueError, match="identity"):
        encode_handshake_frame("too-long-identity", 0, 0)
    with pytest.raises(ValueError, match="nonce"):
        encode_handshake_frame("dev-0001", 256, 0)
    with pytest.raises(ValueError, match="mcs"):
        encode_handshake_frame("dev-0001", 0, 4)


def test_happy_path() -> None:
    out = drive(
        (HandshakeEvent.ARM, 0.0),
        (HandshakeEvent.PREAMBLE_DETECTED, 0.05),
        (HandshakeEvent.FRAME_OK, 0.20),
        (HandshakeEvent.ACK_SENT, 0.25),
        (HandshakeEvent.DATA_SENT, 0.30),
        (HandshakeEvent.DATA_FINISHED, 0.90),
    )
    assert out.state.state is HandshakeStateName.IDLE
    assert out.state.retries == 0
    # Spot-check the intermediate states by walking again:
    s = initial_state()
    s = step(s, HandshakeEvent.ARM, 0.0).state
    assert s.state is HandshakeStateName.LISTENING
    out = step(s, HandshakeEvent.PREAMBLE_DETECTED, 0.05)
    assert out.state.state is HandshakeStateName.HANDSHAKE
    assert out.action == "request_tx"
    out = step(out.state, HandshakeEvent.FRAME_OK, 0.20)
    assert out.state.state is HandshakeStateName.READY
    out = step(out.state, HandshakeEvent.ACK_SENT, 0.25)
    assert out.state.state is HandshakeStateName.DATA
    assert out.action == "request_tx"


def test_timeout_retries_then_fails() -> None:
    cfg = HandshakeConfig(max_retries=2)
    s = initial_state()
    s = step(s, HandshakeEvent.ARM, 0.0, config=cfg).state
    # Retry 1: LISTENING -> IDLE (budget 2, consumed 1)
    out = step(s, HandshakeEvent.TIMEOUT, 0.5, config=cfg)
    assert out.state.state is HandshakeStateName.IDLE
    assert out.state.retries == 1
    assert out.action == "retry"
    s = step(out.state, HandshakeEvent.ARM, 0.6, config=cfg).state
    # Retry 2: LISTENING -> IDLE (budget exhausted)
    out = step(s, HandshakeEvent.TIMEOUT, 1.1, config=cfg)
    assert out.state.state is HandshakeStateName.IDLE
    assert out.state.retries == 2
    s = step(out.state, HandshakeEvent.ARM, 1.2, config=cfg).state
    # Retry 3 exceeds budget -> FAILED + abort
    out = step(s, HandshakeEvent.TIMEOUT, 1.7, config=cfg)
    assert out.state.state is HandshakeStateName.FAILED
    assert out.action == "abort"
    # Reset recovers to IDLE with a cleared retry budget.
    out = step(out.state, HandshakeEvent.RESET, 2.0, config=cfg)
    assert out.state.state is HandshakeStateName.IDLE
    assert out.state.retries == 0


def test_corrupt_frame_retries() -> None:
    cfg = HandshakeConfig(max_retries=1)
    s = initial_state()
    s = step(s, HandshakeEvent.ARM, 0.0, config=cfg).state
    s = step(s, HandshakeEvent.PREAMBLE_DETECTED, 0.1, config=cfg).state
    out = step(s, HandshakeEvent.FRAME_CORRUPT, 0.3, config=cfg)
    assert out.state.state is HandshakeStateName.LISTENING
    assert out.state.retries == 0  # FRAME_CORRUPT does not burn the budget
    out = step(out.state, HandshakeEvent.TIMEOUT, 0.8, config=cfg)
    assert out.state.state is HandshakeStateName.IDLE
    assert out.state.retries == 1


def test_invalid_events_raise() -> None:
    s = initial_state()
    with pytest.raises(HandshakeError, match="not valid"):
        step(s, HandshakeEvent.PREAMBLE_DETECTED, 0.0)  # must ARM first
    with pytest.raises(HandshakeError, match="not valid"):
        step(s, HandshakeEvent.FRAME_OK, 0.0)
    s = step(s, HandshakeEvent.ARM, 0.0).state
    with pytest.raises(HandshakeError, match="not valid"):
        step(s, HandshakeEvent.DATA_FINISHED, 0.1)


def test_time_must_be_monotonic() -> None:
    s = initial_state()
    s = step(s, HandshakeEvent.ARM, 5.0).state
    with pytest.raises(HandshakeError, match="before last event"):
        step(s, HandshakeEvent.PREAMBLE_DETECTED, 4.0)


def test_step_is_pure() -> None:
    before = initial_state()
    out = step(before, HandshakeEvent.ARM, 0.0)
    # Input is untouched (frozen dataclass, but assert anyway) and output is
    # a distinct object with the updated state.
    assert before.state is HandshakeStateName.IDLE
    assert out.state is not before
    assert out.state.state is HandshakeStateName.LISTENING


def test_transitions_table_covers_all_states() -> None:
    states = set(HandshakeStateName)
    covered = {state for state, _ in TRANSITIONS}
    # FAILED is terminal-by-budget only (reachable, no outgoing transitions
    # except RESET is a design choice asserted via the table below).
    for (state, event), (nstate, action) in TRANSITIONS.items():
        assert state in states
        assert event in set(HandshakeEvent)
        assert nstate in states
        assert action in {"none", "request_tx", "retry"}
    # Every non-FAILED state has at least one outgoing transition.
    assert states - covered <= {HandshakeStateName.FAILED}
    assert len(TRANSITIONS) > 0
