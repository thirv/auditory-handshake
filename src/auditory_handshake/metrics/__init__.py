"""Information-theoretic and link metrics.

Capacity ceiling (``capacity.py``, S3) implements the Shannon bound and
water-filling allocation. RF baselines live in ``models/``; BER/BLER/goodput
arrive with the M3–M9 waves.

The perceptual audibility score (``audibility.py``) implements the C2
degeneracy criterion: ``audibility_cost_db(f, L) = L + A(f) - threshold(f)``,
with a pluggable A-weighting source (librosa preferred, native IEC 6167
fallback) and a Fletcher-Munson 0-phon hearing-threshold table.

Channel metrics (``channel_metrics.py``, S10.1) add RT60 (Schroeder integral),
``tau_mean``, ``tau_rms`` and ``B50``/``B90`` as pure NumPy functions on RIRs.

Perceptual metrics (``perceptual.py``) add ZWTV loudness, DIN-45637 sharpness
and roughness via the ``mosqito`` package (imported lazily on first call).

The public names above are re-exported lazily (PEP 562): importing this package
does not eagerly load the submodules, and each name is resolved on first access.
This keeps ``import auditory_handshake.metrics`` light and silences the
"found in sys.modules ... prior to execution" warning when a demo is run with
``python -m``.
"""

# Public names re-exported from their source submodules, resolved lazily via
# PEP 562 (__getattr__ below). Importing a submodule directly (e.g.
# ``metrics.channel_metrics``) is always available. Keeping the re-exports
# lazy means ``import auditory_handshake.metrics`` no longer eagerly loads every
# submodule, and running a demo with ``python -m`` no longer trips CPython's
# "found in sys.modules ... prior to execution" warning (the parent import was
# re-running an already-imported submodule).
#
# The channel_metrics names were previously listed in ``__all__`` without an
# import, so ``from auditory_handshake.metrics import rt60_schroeder`` raised
# ImportError; the lazy map below makes them resolve as documented.
_LAZY_EXPORTS: dict[str, str] = {
    # Link budget
    "link_budget": "link_budget",
    # Channel metrics (S10.1)
    "rt60_schroeder": "channel_metrics",
    "tau_mean": "channel_metrics",
    "tau_rms": "channel_metrics",
    "b50": "channel_metrics",
    "b90": "channel_metrics",
    # TTCH (handshake time) model - PLAN.md §2.4
    "ttch_rtt_s": "ttch",
    "classify_regime": "ttch",
    "regress_rt60_sensitivity": "ttch",
    # TTFB (time to first bit) — M4 detection-latency metrics
    "TTCH_TTFB_TARGET_MS": "ttfb",
    "TTFBReport": "ttfb",
    "compute_ttfb": "ttfb",
    # Two-way handshake round-trip — C1 empirical close-out
    "TwoWayLeg": "two_way",
    "rtt_s": "two_way",
    "regress_ttch_vs_rt60": "two_way",
    # Perceptual audibility score (C2 criterion)
    "a_weighting_db": "audibility",
    "hearing_threshold_db": "audibility",
    "audibility_cost_db": "audibility",
    # Goodput–audibility trade-off surface (S17.1)
    "DesignPoint": "rate",
    "LlambdaTrace": "rate",
    "gross_bit_rate_bps": "rate",
    "goodput_bps": "rate",
    "aggregate_audibility_cost_db": "rate",
    "normalized_utility": "rate",
    "nondominated": "rate",
    "trace_optimum_over_lambda": "rate",
    "c2_verdict": "rate",
    # Perceptual acoustic metrics (mosqito): loudness / sharpness / roughness
    "PerceptualMetrics": "perceptual",
    "perceptual_metrics": "perceptual",
    "noise_source_perceptual_metrics": "perceptual",
    "PerceptualAnnoyance": "perceptual",
    "perceptual_annoyance": "perceptual",
}


def __getattr__(name: str) -> object:
    """Lazily import and re-export a submodule attribute (PEP 562).

    Lets ``from auditory_handshake.metrics import link_budget`` (and the
    channel/TTCH/audibility helpers) resolve without loading every submodule
    at package import time.
    """
    import importlib

    submodule = _LAZY_EXPORTS.get(name)
    if submodule is not None:
        return getattr(importlib.import_module(f".{submodule}", __name__), name)
    # A submodule imported via the package (e.g. ``metrics.channel_metrics``)
    # resolves on the first attribute access; the second is then found in
    # sys.modules by the import system, not here.
    if name in (
        "audibility",
        "channel_metrics",
        "link_budget",
        "rate",
        "two_way",
        "ttch",
        "ttfb",
        "perceptual",
    ):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "link_budget",
    "rt60_schroeder",
    "tau_mean",
    "tau_rms",
    "b50",
    "b90",
    # TTCH (handshake time) model - PLAN.md §2.4
    "ttch",
    "ttch_rtt_s",
    "classify_regime",
    "regress_rt60_sensitivity",
    # TTFB (time to first bit) — M4 detection-latency metrics
    "ttfb",
    "TTCH_TTFB_TARGET_MS",
    "TTFBReport",
    "compute_ttfb",
    # Two-way handshake round-trip — C1 empirical close-out
    "two_way",
    "TwoWayLeg",
    "rtt_s",
    "regress_ttch_vs_rt60",
    # Perceptual audibility score (C2 criterion)
    "a_weighting_db",
    "hearing_threshold_db",
    "audibility_cost_db",
    # Goodput–audibility trade-off surface (S17.1)
    "DesignPoint",
    "LlambdaTrace",
    "gross_bit_rate_bps",
    "goodput_bps",
    "aggregate_audibility_cost_db",
    "normalized_utility",
    "nondominated",
    "trace_optimum_over_lambda",
    "c2_verdict",
    # Perceptual acoustic metrics (mosqito)
    "PerceptualMetrics",
    "perceptual_metrics",
    "noise_source_perceptual_metrics",
    "PerceptualAnnoyance",
    "perceptual_annoyance",
]
