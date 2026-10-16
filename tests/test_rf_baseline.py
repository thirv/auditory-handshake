"""Tests for the cited RF baseline latency table (S5.1, PLAN.md §1.3).

Invariants:

- the table is PLAN.md §1.3 row-for-row (paths, ranges, order);
- ``[V]`` rows are structurally unquotable until S5.2 closes them;
- the two "honest hard baselines" are flagged and are exactly two;
- the acoustic rows are this work's targets, never marked quotable;
- ``render_table`` shows the ``[V]`` marker on every unverified RF row.
"""

from __future__ import annotations

import dataclasses

import pytest

from auditory_handshake.models.rf_baseline import (
    BASELINES,
    SourceStatus,
    VerificationStatus,
    hard_baselines,
    render_table,
)

# --- Table fidelity vs PLAN.md §1.3 -----------------------------------------


def test_table_has_nine_rows_in_plan_order() -> None:
    paths = [e.path for e in BASELINES]
    assert paths == [
        "802.11 open auth + assoc",
        "+ WPA2 4-way + DHCP",
        "Wi-Fi, measured from user intent",
        "BLE connect attempt",
        "BLE advertisement reception (no connect)",
        "WiFi monitor-mode frame RX (no assoc)",
        "Acoustic preamble detect + header parse",
        'Acoustic one-shot "shout" frame',
        "Acoustic 2-way handshake",
    ]


def test_latency_ranges_match_plan_section_1_3() -> None:
    expected = {
        "802.11 open auth + assoc": (30.0, 100.0),
        "+ WPA2 4-way + DHCP": (150.0, 600.0),
        "Wi-Fi, measured from user intent": (1000.0, 3000.0),
        "BLE connect attempt": (1000.0, 3000.0),
        "BLE advertisement reception (no connect)": (3.0, 30.0),
        "WiFi monitor-mode frame RX (no assoc)": (1.0, 10.0),
        "Acoustic preamble detect + header parse": (6.0, 8.0),
        'Acoustic one-shot "shout" frame': (10.0, 25.0),
        "Acoustic 2-way handshake": (30.0, 120.0),
    }
    for e in BASELINES:
        assert e.latency_ms == expected[e.path]


# --- Honesty discipline ([V] / quotability) ----------------------------------


def test_wifi_assoc_and_wpa2_rows_now_verified() -> None:
    """S5.2 closed: the two Wi-Fi auth/assoc rows are VERIFIED + quotable."""
    for path in ("802.11 open auth + assoc", "+ WPA2 4-way + DHCP"):
        e = next(x for x in BASELINES if x.path == path)
        assert e.verification is VerificationStatus.VERIFIED
        assert e.figure_sourced is True  # verified primary states the ms figure
        assert e.quotable
        assert not e.v_marker
        ids = {s.identifier for s in e.sources}
        # Both verified anchors cited (arXiv + Crossref-confirmed DOI).
        assert "arXiv:0304054" in ids  # Mishra, Banerjee & Pang 2003
        assert "10.1109/infcom.2005.1497933" in ids  # Ramani et al. 2005 (SyncScan)


def test_wifi_user_intent_row_still_unverified() -> None:
    """The UX-level "user intent" row has no verified primary — [V] stands."""
    e = next(x for x in BASELINES if x.path == "Wi-Fi, measured from user intent")
    assert e.verification is VerificationStatus.UNVERIFIED
    assert e.figure_sourced is False
    assert not e.quotable
    assert e.v_marker


def test_wifi_monitor_rx_row_still_unverified() -> None:
    """Monitor-mode RX row has NO verified primary — [V] must stand (S5.2 open)."""
    e = next(x for x in BASELINES if x.path == "WiFi monitor-mode frame RX (no assoc)")
    assert e.verification is VerificationStatus.UNVERIFIED
    assert e.figure_sourced is False
    assert not e.quotable
    assert e.v_marker
    assert e.hard  # honest hard baseline stays flagged


def test_ble_adv_row_verified_metadata_but_figure_still_unsourced() -> None:
    e = next(x for x in BASELINES if x.path == "BLE advertisement reception (no connect)")
    # Both sources now VERIFIED (Gómez 2012; Shen 2023 via arXiv API 2026-09-28).
    # Metadata is closed, but the 3–30 ms figure has no primary numeric source,
    # so the row is kept unquotable as fact (RELATED_WORK.md §5).
    assert e.verification is VerificationStatus.VERIFIED
    assert e.figure_sourced is False  # metadata verified, but figure not sourced
    assert not e.quotable


def test_ble_connect_row_is_verified() -> None:
    e = next(x for x in BASELINES if x.path == "BLE connect attempt")
    assert e.verification is VerificationStatus.VERIFIED
    assert e.figure_sourced is True  # the one RF row whose figure is sourced
    assert e.quotable


def test_no_verified_row_without_sources() -> None:
    for e in BASELINES:
        if e.verification is VerificationStatus.VERIFIED:
            assert len(e.sources) > 0
            assert all(s.status is SourceStatus.VERIFIED for s in e.sources)


# --- Hard baselines -----------------------------------------------------------


def test_exactly_two_hard_baselines() -> None:
    hard = hard_baselines()
    assert [e.path for e in hard] == [
        "BLE advertisement reception (no connect)",
        "WiFi monitor-mode frame RX (no assoc)",
    ]


# --- This-work rows ------------------------------------------------------------


def test_acoustic_rows_are_this_work_and_never_quotable() -> None:
    for e in BASELINES:
        if e.path.startswith("Acoustic"):
            assert e.this_work
            assert e.verification is VerificationStatus.THIS_WORK
            assert not e.quotable
            assert not e.v_marker


# --- Sources / rendering -------------------------------------------------------


def test_source_identifiers_match_related_work() -> None:
    gomez = next(s for s in BASELINES[3].sources if "Gómez" in s.authors)
    assert gomez.identifier == "10.3390/s120911734"
    assert gomez.status is SourceStatus.VERIFIED
    shen = next(s for s in BASELINES[4].sources if "Shen" in s.authors)
    assert shen.identifier == "arXiv:2312.15650"
    assert shen.status is SourceStatus.VERIFIED


def test_render_table_marks_v_rows_and_hard_baselines() -> None:
    table = render_table()
    assert "[V] = no verified primary source yet" in table
    assert "802.11 open auth + assoc" in table

    # S5.2 closed for the two verified Wi-Fi rows: they render WITHOUT a [V] marker.
    for row in ("~30–100 ms", "~150–600 ms"):
        line = next(candidate for candidate in table.splitlines() if row in candidate)
        assert " [V]" not in line
        assert "verified" in line

    # Unverified Wi-Fi rows still carry the [V] marker.
    assert "1–3 s [V]" in table
    assert "~1–10 ms [V]" in table
    assert "WiFi monitor-mode frame RX (no assoc)" in table

    # hard-baseline marker terminates exactly two rows (the two honest hard baselines)
    assert sum(1 for line in table.splitlines() if line.endswith("  *")) == 2


def test_render_table_accepts_subset() -> None:
    subset = tuple(e for e in BASELINES if e.hard)
    assert render_table(subset).count("path") == 1


def test_entry_fields_are_frozen() -> None:
    e = BASELINES[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.note = "mutated"  # type: ignore[misc]
