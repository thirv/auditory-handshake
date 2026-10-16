"""Tests for the perceptual audibility score (C2 criterion).

C2 requirements under test: the score is **band-dependent**, **monotone in
SPL**, and **non-degenerate** (distinct bands -> distinct costs). Also the
pluggable A-weighting (librosa preferred, native IEC 6167 fallback).
"""

from __future__ import annotations

import numpy as np
import pytest

from auditory_handshake.metrics.audibility import (
    _a_weighting_native,
    _resolve_a_weighting_backend,
    a_weighting_db,
    audibility_cost_db,
    hearing_threshold_db,
)

try:  # pragma: no cover - librosa is optional
    import librosa  # noqa: F401

    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    _HAS_LIBROSA = False


def _a(f: float) -> float:
    return float(a_weighting_db([f])[0])


class TestAWeighting:
    def test_anchor_1khz_is_zero(self) -> None:
        # librosa (and the identical native fallback) hit ~0 dB at 1 kHz;
        # the formula is a log-domain approximation so it's off by <1e-3 dB.
        assert _a(1000.0) == pytest.approx(0.0, abs=0.01)

    def test_known_reference_values(self) -> None:
        # Values from librosa.A_weighting (reproduced exactly by the native
        # IEC 6167 fallback), so the dependency stays optional.
        assert _a(2000.0) == pytest.approx(1.2018, abs=0.01)
        assert _a(500.0) == pytest.approx(-3.2473, abs=0.01)
        assert _a(20000.0) == pytest.approx(-9.3466, abs=0.01)

    def test_shape_low_and_high_roll_off(self) -> None:
        # Mid-band is the reference; both ends are attenuated.
        assert _a(100.0) < _a(1000.0)
        assert _a(10000.0) < _a(1000.0)

    def test_vectorised(self) -> None:
        out = a_weighting_db(np.array([1000.0, 2000.0, 10000.0]))
        assert out.shape == (3,)
        assert out[0] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.skipif(not _HAS_LIBROSA, reason="librosa not installed")
    def test_native_matches_librosa_exactly(self) -> None:
        f = np.array([20.0, 100.0, 1000.0, 4000.0, 16000.0, 25000.0])
        native = _a_weighting_native(f)
        ref = np.asarray(librosa.A_weighting(f, min_db=None), dtype=float)
        assert float(np.max(np.abs(native - ref))) < 1e-9

    def test_fallback_to_native_when_librosa_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from auditory_handshake.metrics import audibility as mod

        class _BrokenImport:
            def import_module(self, *_a: object, **_k: object) -> object:
                raise ModuleNotFoundError("librosa")

        monkeypatch.setattr(mod, "importlib", _BrokenImport())
        monkeypatch.setattr(mod, "_A_WEIGHTING_BACKEND", None)
        assert _resolve_a_weighting_backend() is _a_weighting_native


class TestHearingThreshold:
    def test_anchor_1khz_is_zero(self) -> None:
        assert float(hearing_threshold_db(1000.0)) == pytest.approx(0.0)

    def test_midband_is_flat(self) -> None:
        assert float(hearing_threshold_db(500.0)) == pytest.approx(0.0, abs=0.5)
        assert float(hearing_threshold_db(2000.0)) == pytest.approx(0.0, abs=0.5)

    def test_edges_rise(self) -> None:
        # Near-ultrasonic and low-frequency both cost more SPL to hear.
        assert float(hearing_threshold_db(20000.0)) > 10.0
        assert float(hearing_threshold_db(30.0)) > 5.0

    def test_out_of_range_holds_endpoints(self) -> None:
        assert float(hearing_threshold_db(10.0)) == pytest.approx(float(hearing_threshold_db(20.0)))
        assert float(hearing_threshold_db(50000.0)) == pytest.approx(
            float(hearing_threshold_db(25000.0))
        )

    def test_invalid_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            hearing_threshold_db(-5.0)


class TestAudibilityCost:
    def test_1khz_equals_level(self) -> None:
        # At 1 kHz A(f)~0 and threshold(f)=0, so cost ~ SPL.
        assert audibility_cost_db(1000.0, 80.0) == pytest.approx(80.0, abs=0.01)

    def test_reference_value_2khz(self) -> None:
        # 80 dB SPL @ 2 kHz: 80 + A(2 kHz) - threshold(2 kHz) = 80 + 1.2018.
        assert audibility_cost_db(2000.0, 80.0) == pytest.approx(81.20, abs=0.01)

    def test_monotone_in_spl(self) -> None:
        # C2: strictly increasing in level for a fixed band.
        lo, mid, hi = (audibility_cost_db(1500.0, lvl) for lvl in (60.0, 80.0, 100.0))
        assert lo < mid < hi
        assert hi - mid == pytest.approx(mid - lo)  # linear in SPL

    def test_band_dependent_non_degenerate(self) -> None:
        # C2: at a fixed SPL, distinct bands give distinct costs.
        bands = (100.0, 500.0, 1000.0, 2000.0, 8000.0, 20000.0)
        costs = [audibility_cost_db(b, 80.0) for b in bands]
        assert len(set(costs)) == len(bands)

    def test_near_ultrasonic_penalised(self) -> None:
        # A 20 kHz tone is far less audible than a 1 kHz tone at equal SPL.
        assert audibility_cost_db(20000.0, 80.0) < audibility_cost_db(1000.0, 80.0)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="frequency_hz"):
            audibility_cost_db(0.0, 80.0)
        with pytest.raises(ValueError, match="level_db"):
            audibility_cost_db(1000.0, float("nan"))
