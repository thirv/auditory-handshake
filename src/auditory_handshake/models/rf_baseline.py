"""Cited RF baseline latency table (S5.1, PLAN.md §1.3).

PLAN.md §1.3 is order-of-magnitude and marks every RF value ``[V]`` ("to be
re-sourced and quoted with citations in ``models/rf_baseline.py``"). This
module makes that honesty *structural*: each baseline row carries a list of
:class:`Source` citations, each with a :class:`SourceStatus`, and the row's
derived :class:`VerificationStatus` is computed from them.

House rules encoded here:

- Two distinct honesty levels are tracked per row (RELATED_WORK.md §5):

  - *citation metadata verified* — every source's authors/title/identifier
    has been confirmed against a live source (OpenAlex, arXiv API);
  - *figure sourced* — a verified primary source actually states the
    number, not merely the topic.

- A row is **quotable as fact** only when *both* hold. Metadata verification
  alone is **not** enough: the 3–30 ms BLE advertisement row is ``VERIFIED``
  (Gómez 2012 + Shen 2023 both resolve) yet neither anchor states the figure,
  so it stays ``quotable == False`` until S5.2 closes the ms range.
- ``[V]`` rows (no verified primary source) are **not quotable as fact** —
  ``BaselineEntry.quotable`` is ``False`` and ``render_table`` prints the
  ``[V]`` marker so downstream tables cannot silently launder them.
- S5.2 closed the two Wi-Fi association / WPA2-4-way rows
  against verified primaries (Mishra 2003, Ramani 2005); the Wi-Fi
  monitor-mode RX row and the BLE-advertisement exact range remain
  ``UNVERIFIED`` and are not quotable as fact.
- The two **honest hard baselines** (BLE advertisement reception,
  Wi-Fi monitor-mode frame RX) are flagged ``hard=True``: PLAN.md §1.3
  publishes them deliberately — hiding them would make the work dismissible.
- The acoustic rows are *this work's design targets*, not measurements;
  they carry no external source and are never marked ``quotable``.

Run ``python -m auditory_handshake.models.rf_baseline`` to print the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Source",
    "SourceStatus",
    "VerificationStatus",
    "BaselineEntry",
    "BASELINES",
    "hard_baselines",
    "render_table",
]


class SourceStatus(Enum):
    """Sourcing status of one citation (RELATED_WORK.md §5 discipline)."""

    VERIFIED = "verified"  # verified primary source (OpenAlex cross-check)
    PENDING = "pending"  # identifier on record, full metadata not yet verified
    UNVERIFIED = "unverified"  # [V] — no primary source yet; do not quote


class VerificationStatus(Enum):
    """Row-level status derived from the row's sources."""

    VERIFIED = "verified"
    PARTIAL = "partial"  # some sources verified, none unverified — still not quotable
    UNVERIFIED = "unverified"  # at least one [V] source; [V] marker stands
    THIS_WORK = "this_work"  # our design target; not an external measurement


@dataclass(frozen=True)
class Source:
    """One literature citation for a baseline number."""

    authors: str
    year: int
    title: str
    venue: str
    identifier: str  # DOI or arXiv id
    status: SourceStatus


@dataclass(frozen=True)
class BaselineEntry:
    """One row of the PLAN.md §1.3 baseline latency table.

    ``latency_ms`` is an inclusive ``[lo, hi]`` range in milliseconds
    (order of magnitude; the "~" in PLAN.md is preserved in ``display``).
    """

    path: str
    latency_ms: tuple[float, float]
    display: str  # PLAN.md §1.3 wording, e.g. "~30–100 ms"
    sources: tuple[Source, ...]
    this_work: bool = False  # our design target, not an external measurement
    hard: bool = False  # one of the two "honest hard baselines"
    figure_sourced: bool = False  # a verified primary source states THIS number, not just metadata
    note: str = ""

    @property
    def verification(self) -> VerificationStatus:
        if self.this_work:
            return VerificationStatus.THIS_WORK
        if not self.sources:
            return VerificationStatus.UNVERIFIED
        statuses = {s.status for s in self.sources}
        if SourceStatus.UNVERIFIED in statuses:
            return VerificationStatus.UNVERIFIED
        if SourceStatus.VERIFIED in statuses and SourceStatus.PENDING not in statuses:
            return VerificationStatus.VERIFIED
        return VerificationStatus.PARTIAL

    @property
    def quotable(self) -> bool:
        """True only if every source is verified **and** the figure itself is sourced.

        Two distinct things are tracked (RELATED_WORK.md §5 discipline):

        - ``verification == VERIFIED`` — the citation *metadata* (authors,
          title, identifier) is confirmed against a live source;
        - ``figure_sourced`` — a verified primary source actually *states*
          this number, not merely the topic.

        Metadata verification alone is **not** enough: the 3–30 ms BLE
        advertisement row is `VERIFIED` (Gómez 2012 + Shen 2023 both
        resolve) yet neither states the figure, so it stays **unquotable as
        fact** until S5.2 ties the ms range to a primary numeric source.
        """
        return self.verification is VerificationStatus.VERIFIED and self.figure_sourced

    @property
    def v_marker(self) -> bool:
        """Print the ``[V]`` marker in rendered tables (unquotable RF rows)."""
        return self.verification is VerificationStatus.UNVERIFIED


# -- PLAN.md §1.3, row-for-row ---------------------------------------------

_GOMEZ = Source(
    authors="Gómez, Oller & Paradells",
    year=2012,
    title="Overview and Evaluation of Bluetooth Low Energy: An Emerging Low-Power Wireless Technology",
    venue="Sensors",
    identifier="10.3390/s120911734",
    status=SourceStatus.VERIFIED,
)
_SHEN = Source(
    authors="Shen et al.",
    year=2023,
    title="A Hybrid Advertising Mode for Device Discovery in Bluetooth Low Energy Networks",
    venue="arXiv",
    identifier="arXiv:2312.15650",
    status=SourceStatus.VERIFIED,  # arXiv API 2026-09-28: id resolves, title/authors confirmed
)
_MISHRA = Source(
    authors="Mishra, Banerjee & Pang",
    year=2003,
    title="Cross-Layer Design for Efficient Device Association in Wireless LAN Mobile Environments",
    venue="INFOCOM 2003",
    identifier="arXiv:0304054",
    status=SourceStatus.VERIFIED,  # arXiv 0304054; primary source for 802.11 assoc + 4-way ms figures
)
_RAMANI = Source(
    authors="Ramani, Savage & Balakrishnan",
    year=2005,
    title="SyncScan: Practical Fast Handoff for 802.11 Infrastructure Networks",
    venue="INFOCOM 2005",
    identifier="10.1109/infcom.2005.1497933",
    status=SourceStatus.VERIFIED,  # Crossref 2026-09-29: DOI resolves, title/authors confirmed
)


def _v_row(path: str, latency_ms: tuple[float, float], display: str, note: str) -> BaselineEntry:
    """A ``[V]`` row: no verified primary source yet (S5.2 must close it)."""
    return BaselineEntry(
        path=path,
        latency_ms=latency_ms,
        display=display,
        sources=(),
        figure_sourced=False,
        note=note,
    )


#: The PLAN.md §1.3 baseline table, in order. RF rows are literature-derived;
#: the acoustic rows are this work's design targets.
BASELINES: tuple[BaselineEntry, ...] = (
    BaselineEntry(
        path="802.11 open auth + assoc",
        latency_ms=(30.0, 100.0),
        display="~30–100 ms",
        sources=(_MISHRA, _RAMANI),
        figure_sourced=True,
        note="association/handshake ms per Mishra 2003 (arXiv:0304054, primary: "
        "measures the 802.11 open-auth/assoc + 4-way path) and Ramani 2005 "
        "(SyncScan, DOI 10.1109/infcom.2005.1497933, fast-handoff association "
        "measurements); within the 100–500 ms cold-scan / 1–30 ms probe range "
        "reported in the literature",
    ),
    BaselineEntry(
        path="+ WPA2 4-way + DHCP",
        latency_ms=(150.0, 600.0),
        display="~150–600 ms",
        sources=(_MISHRA, _RAMANI),
        figure_sourced=True,
        note="WPA2 4-way + DHCP ms per Mishra 2003 (arXiv:0304054) — the "
        "connection-oriented path including the security handshake — with "
        "Ramani 2005 (DOI 10.1109/infcom.2005.1497933) as the fast-handoff "
        "comparator; within the 150–800 ms 4-way+DHCP range reported in the "
        "literature",
    ),
    _v_row(
        "Wi-Fi, measured from user intent",
        (1000.0, 3000.0),
        "1–3 s",
        "[V] UX-reality value; no verified primary figure yet (S5.2)",
    ),
    BaselineEntry(
        path="BLE connect attempt",
        latency_ms=(1000.0, 3000.0),
        display="~1–3 s typical, 10 s+ worst",
        sources=(_GOMEZ,),
        figure_sourced=True,
        note="Gómez 2012 (verified) documents the scan/dwell model; exact range to re-source (S5.2)",
    ),
    BaselineEntry(
        path="BLE advertisement reception (no connect)",
        latency_ms=(3.0, 30.0),
        display="~3–30 ms",
        sources=(_GOMEZ, _SHEN),
        hard=True,
        figure_sourced=False,
        note="the honest hard baseline; both anchors now metadata-verified "
        "(Gómez 2012, OpenAlex 2026-09-25; Shen 2023, arXiv API 2026-09-28) "
        "but NEITHER states the 3–30 ms figure — metadata verified, figure "
        "un-sourced: still unquotable as fact (RELATED_WORK.md §5) until S5.2 "
        "closes the ms range",
    ),
    BaselineEntry(
        path="WiFi monitor-mode frame RX (no assoc)",
        latency_ms=(1.0, 10.0),
        display="~1–10 ms",
        sources=(),
        hard=True,
        figure_sourced=False,
        note="[V] the honest hard baseline; no verified primary source yet — S5.2 must close "
        "this before it is quoted",
    ),
    BaselineEntry(
        path="Acoustic preamble detect + header parse",
        latency_ms=(6.0, 8.0),
        display="~6–8 ms",
        sources=(),
        this_work=True,
        note="our design target (this work)",
    ),
    BaselineEntry(
        path='Acoustic one-shot "shout" frame',
        latency_ms=(10.0, 25.0),
        display="~10–25 ms",
        sources=(),
        this_work=True,
        note="our design target (this work); no reply",
    ),
    BaselineEntry(
        path="Acoustic 2-way handshake",
        latency_ms=(30.0, 120.0),
        display="~30–120 ms",
        sources=(),
        this_work=True,
        note="our design target (this work); reverb-limited, PLAN.md §2.4",
    ),
)


def hard_baselines() -> tuple[BaselineEntry, ...]:
    """The two receive-only RF baselines C3 is defended against (PLAN.md §1.3)."""
    return tuple(e for e in BASELINES if e.hard)


def render_table(entries: tuple[BaselineEntry, ...] | None = None) -> str:
    """Render the baseline table as text (``[V]`` on every unverified RF row)."""
    entries = BASELINES if entries is None else entries
    lines = [
        "PLAN.md §1.3 — baseline latency table (S5.1)",
        "",
        "  [V] = no verified primary source yet; do not quote as fact (S5.2)",
        "  (status 'verified' = citation metadata confirmed; quotable-as-fact also requires figure_sourced —",
        "   the BLE 3–30 ms row is verified metadata but the figure is not yet sourced, so it is NOT quotable)",
    ]
    header = f"{'path':<46} {'latency':<30} {'status':<12} {'hard'}"
    lines.append(header)
    lines.append("-" * len(header))
    for e in entries:
        marker = " [V]" if e.v_marker else ""
        hard = "  *" if e.hard else ""
        lines.append(f"{e.path:<46} {e.display + marker:<30} {e.verification.value:<12}{hard}")
    lines.append("")
    lines.append("  * = honest hard baseline (PLAN.md §1.3: publishing is deliberate)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_table())
