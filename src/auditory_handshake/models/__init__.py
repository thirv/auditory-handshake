"""Reference models for comparison baselines.

``rf_baseline.py`` (S5) holds the cited RF connection-establishment numbers
that the C3 framing claim is defended against.
"""

from auditory_handshake.models.rf_baseline import (
    BASELINES,
    BaselineEntry,
    Source,
    SourceStatus,
    VerificationStatus,
    hard_baselines,
    render_table,
)

__all__ = [
    "BASELINES",
    "BaselineEntry",
    "Source",
    "SourceStatus",
    "VerificationStatus",
    "hard_baselines",
    "render_table",
]
