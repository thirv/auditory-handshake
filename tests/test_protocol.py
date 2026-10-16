"""Tests for the Phase C handshake protocol primitives (protocol.py).

Covers: the CRC-16/CCITT known-vector check, handshake frame
encode/decode round-trip, CRC-failure and length validation, and the
identity / nonce / MCS field validation rules.
"""

from __future__ import annotations

import dataclasses

import pytest

from auditory_handshake.protocol import (
    HANDSHAKE_FRAME_LEN,
    HandshakeFrame,
    crc16_ccitt,
    decode_handshake_frame,
    encode_handshake_frame,
)

# Fixed identity so tests are stable and readable.
IDENTITY = b"dev-0001"


# --- CRC-16/CCITT --------------------------------------------------------


def test_crc16_known_vector() -> None:
    """CRC-16/CCITT-FALSE of b'123456789' must equal 0x29B1."""
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_crc16_empty_is_init() -> None:
    assert crc16_ccitt(b"") == 0xFFFF


def test_crc16_depends_on_all_bytes() -> None:
    assert crc16_ccitt(b"abc") != crc16_ccitt(b"abd")
    # Order matters — different permutation, different CRC.
    assert crc16_ccitt(b"abc") != crc16_ccitt(b"cba")


# --- Frame round-trip ----------------------------------------------------


def test_encode_decode_roundtrip() -> None:
    raw = encode_handshake_frame(IDENTITY, nonce=42, mcs=2)
    assert len(raw) == HANDSHAKE_FRAME_LEN == 12

    frame = decode_handshake_frame(raw)
    assert frame.identity == IDENTITY
    assert frame.nonce == 42
    assert frame.mcs == 2


def test_encode_decode_roundtrip_extremes() -> None:
    for nonce, mcs in [(0, 0), (255, 3), (0, 3), (255, 0)]:
        frame = decode_handshake_frame(encode_handshake_frame(IDENTITY, nonce, mcs))
        assert (frame.identity, frame.nonce, frame.mcs) == (IDENTITY, nonce, mcs)


def test_frame_to_bytes_matches_encode() -> None:
    frame = decode_handshake_frame(encode_handshake_frame(IDENTITY, 7, 1))
    assert frame.to_bytes() == encode_handshake_frame(IDENTITY, 7, 1)


def test_frame_to_dict() -> None:
    frame = HandshakeFrame(identity=b"abc123xy", nonce=1, mcs=0)
    assert frame.to_dict() == {"identity": "abc123xy", "nonce": 1, "mcs": 0}


# --- Invalid frames ------------------------------------------------------


def test_decode_rejects_bad_length() -> None:
    for bad_len in (0, 11, 13, 100):
        with pytest.raises(ValueError, match="12 bytes"):
            decode_handshake_frame(b"\x00" * bad_len)


def test_decode_rejects_corrupt_crc() -> None:
    raw = bytearray(encode_handshake_frame(IDENTITY, 0, 0))
    raw[0] ^= 0x01  # flip one bit in the identity
    with pytest.raises(ValueError, match="CRC mismatch"):
        decode_handshake_frame(bytes(raw))

    # Corrupt a CRC byte instead — body intact, trailing bytes wrong.
    raw = bytearray(encode_handshake_frame(IDENTITY, 5, 2))
    raw[-1] ^= 0xFF
    with pytest.raises(ValueError, match="CRC mismatch"):
        decode_handshake_frame(bytes(raw))


# --- Field validation ----------------------------------------------------


def test_encode_rejects_bad_identity() -> None:
    with pytest.raises(ValueError, match="8 bytes"):
        encode_handshake_frame(b"short", 0, 0)
    with pytest.raises(ValueError, match="8 bytes"):
        encode_handshake_frame("toolongidentity", 0, 0)


def test_encode_rejects_bad_nonce() -> None:
    for bad in (-1, 256, True, 2.0):
        with pytest.raises(ValueError, match="nonce"):
            encode_handshake_frame(IDENTITY, bad, 0)  # type: ignore[arg-type]


def test_encode_rejects_bad_mcs() -> None:
    for bad in (-1, 4, True, "2"):
        with pytest.raises(ValueError, match="mcs"):
            encode_handshake_frame(IDENTITY, 0, bad)  # type: ignore[arg-type]


def test_string_identity_is_utf8_encoded() -> None:
    raw = encode_handshake_frame("dev-0001", 9, 3)
    frame = decode_handshake_frame(raw)
    assert frame.identity == b"dev-0001"


def test_handshake_frame_is_immutable() -> None:
    frame = HandshakeFrame(identity=IDENTITY, nonce=0, mcs=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.nonce = 1  # type: ignore[misc]
