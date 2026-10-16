"""Base dataclasses for the metrics package.

This module is intentionally dependency-free (only the stdlib ``dataclasses``
and ``typing``), so that perceptual-metric result types can be imported from
``auditory_handshake.metrics`` without importing the perceptual backend
(``mosqito``).  See :mod:`auditory_handshake.metrics.perceptual` for the
implementations that populate these types.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PerceptualMetrics", "PerceptualAnnoyance"]


@dataclass(frozen=True)
class PerceptualMetrics:
    """Perceptual acoustic metrics for a single signal (ISO 532 / DIN 45631).

    Attributes
    ----------
    loudness_sone:
        Perceived loudness in *sones* (ISO 532).
    sharpness_acum:
        Sharpness in *acum* (DIN 45631).
    roughness_asper:
        Roughness in *asper* (Daniel & Weber model).

    Notes
    -----
    The dataclass is *frozen* so an instance is an immutable value (it can be
    compared, hashed, and shared safely).  Use :meth:`as_dict` for a plain
    ``{field: value}`` mapping for display or serialization.
    """

    loudness_sone: float
    sharpness_acum: float
    roughness_asper: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain ``{field: value}`` mapping."""
        return {
            "loudness_sone": self.loudness_sone,
            "sharpness_acum": self.sharpness_acum,
            "roughness_asper": self.roughness_asper,
        }


@dataclass(frozen=True)
class PerceptualAnnoyance:
    """A single scalar perceptual-annoyance cost derived from a signal.

    This is the C2-frontier "cost" that replaces the parametric audibility
    term: instead of asking *how loud* a band is in dB, it asks *how annoying
    the actual transmitted sound is perceived to be* — combining loudness,
    sharpness and roughness into one monotone scalar.

    Attributes
    ----------
    N_5:
        95th-percentile instantaneous loudness (sone) — peak loudness,
        reported for context.
    N_avg:
        Mean instantaneous loudness (sone); the scale factor of the cost.
    S_avg:
        Mean instantaneous sharpness (acum).
    R_avg:
        Mean instantaneous roughness (asper).
    annoyance:
        Combined monotone annoyance scalar, ``N_avg * penalty`` with
        ``penalty ∈ [0, 1]`` (0 at/below the tolerable reference, ramping to
        ~1 above it) — hence ``0 ≤ annoyance ≤ N_avg``.  This is the value the
        C2 frontier consumes as its audibility cost.

    Notes
    -----
    Like :class:`PerceptualMetrics`, the dataclass is *frozen* (an immutable
    value) and carries no backend dependency; only :mod:`perceptual` populates
    it.  Use :meth:`as_dict` for a plain mapping.
    """

    N_5: float
    N_avg: float
    S_avg: float
    R_avg: float
    annoyance: float

    def as_dict(self) -> dict[str, float]:
        """Return the annoyance summary as a plain ``{field: value}`` mapping."""
        return {
            "N_5": self.N_5,
            "N_avg": self.N_avg,
            "S_avg": self.S_avg,
            "R_avg": self.R_avg,
            "annoyance": self.annoyance,
        }
