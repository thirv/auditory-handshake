"""S10.1 — Channel metrics on a Room Impulse Response (RIR).

Pure functions (NumPy in, float out; no ``pyroomacoustics`` import) that extract
the channel statistics the PHY design is matched against (PLAN.md §2.2, §7.1
S10/S11; ``configs/phy_constraints.yaml`` schema):

- :func:`rt60_schroeder` — reverberation time from the **Schroeder energy-decay
  integral** ``E(t) = ∫_t^∞ |h(τ)|² dτ``: a least-squares slope of ``log10``
  over a T30-style window (default 5–30 ms), converted to a 60 dB decay time.
- :func:`tau_mean` — mean delay spread of the energy-weighted delay
  distribution.
- :func:`tau_rms` — RMS delay spread (coherence bandwidth and the
  ``τ_rms ≈ 0.205·RT60`` diffuse-field check both key off this).
- :func:`b50` / :func:`b90` — early/late energy fraction times (the time at
  which 50% / 90% of the RIR energy has arrived).

All functions accept a 1-D ``np.ndarray`` (the output of
:func:`auditory_handshake.acoustics.channel.compute_rir`) and a sample rate.
They are pure: same input array in, same float out.

Conventions:

- Delays are time since the **first sample** of the array (the RIR's own
  timeline); callers may trim leading zeros before calling these.
- Non-decaying / anechoic inputs (no negative Schroeder slope, or a
  single-sample impulse) return ``inf`` (or ``0.0`` where the fraction is
  trivially reached) plus a :class:`RuntimeWarning` — callers decide how to
  treat a room with no measurable decay, rather than the metric failing.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "rt60_schroeder",
    "tau_mean",
    "tau_rms",
    "b50",
    "b90",
]

#: T30-style default fit window for :func:`rt60_schroeder` (seconds): the
#: Schroeder integral is flat until ~5 ms (direct sound + early reflections)
#: and unreliable past ~30 ms (noise floor), so the decay slope is fit here.
DEFAULT_RT60_WINDOW_S: tuple[float, float] = (0.005, 0.030)


# --------------------------------------------------------------------------- #
# Shared input validation
# --------------------------------------------------------------------------- #
def _validate_rir_and_fs(
    rir: np.ndarray, fs: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Validate a 1-D RIR and a positive sample rate; return (array, fs, t, e).

    ``e`` is the per-sample energy ``|h|²`` and ``t`` the per-sample delay (s),
    both as float64.
    """
    h = np.asarray(rir)
    if h.ndim != 1:
        raise ValueError(f"rir must be a 1-D array (got shape {h.shape})")
    if h.size == 0:
        raise ValueError("rir must be non-empty")
    h = h.astype(np.float64)
    f = float(fs)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError(f"fs must be > 0 (got {fs})")
    t = np.arange(h.size, dtype=np.float64) / f
    e = h * h
    if not np.all(np.isfinite(e)):
        raise ValueError("rir must contain only finite values")
    return h, f, t, e


# --------------------------------------------------------------------------- #
# RT60 — Schroeder integral
# --------------------------------------------------------------------------- #
def rt60_schroeder(
    rir: np.ndarray,
    fs: float,
    window: tuple[float, float] | None = None,
) -> float:
    """Reverberation time (s) from the Schroeder energy-decay integral.

    The decay curve is the backward energy integral ``E(t) = Σ_{i≥i(t)} |h_i|²``
    (a discrete approximation of ``∫_t^∞ |h(τ)|² dτ``). A least-squares fit of
    ``log10(E)`` versus ``t`` over ``window`` gives the decay rate ``a`` in
    log10 units per second, i.e. ``10·|a|`` dB/s; the 60 dB decay time is
    ``RT60 = 60 / (10·|a|) = 6 / |a|``. For an exponential RIR ``r(t) = e^(−t/λ)``,
    ``E(t) = E₀ e^(−2t/λ)`` and this recovers ``RT60 = 3·ln(10)·λ ≈ 6.908·λ``.

    Parameters
    ----------
    rir:
        1-D room impulse response (real; complex is taken through ``|h|²``).
    fs:
        Sample rate in Hz, matching ``rir``.
    window:
        ``(t0, t1)`` in seconds over which to fit the slope (T30-style; default
        :data:`DEFAULT_RT60_WINDOW_S` = 5–30 ms). ``t1`` is clamped to the RIR
        length; the window must still cover at least two samples.

    Returns
    -------
    float
        RT60 in seconds, or ``inf`` (with a :class:`RuntimeWarning`) if the
        slope in the window is not negative — anechoic / non-decaying input.
    """
    h, f, _t, e = _validate_rir_and_fs(rir, fs)
    if h.size < 2:
        warnings.warn(
            "rt60_schroeder: RIR has < 2 samples — anechoic, returning inf",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf")

    w0, w1 = window if window is not None else DEFAULT_RT60_WINDOW_S
    w0, w1 = float(w0), float(w1)
    if not (np.isfinite(w0) and np.isfinite(w1)) or w0 < 0.0 or w1 <= 0.0 or w0 >= w1:
        raise ValueError(f"window must be (t0, t1) with 0 <= t0 < t1 (got {window})")
    i0 = int(np.clip(round(w0 * f), 0, h.size - 1))
    i1 = int(np.clip(round(w1 * f), 0, h.size - 1))
    if i1 <= i0:
        raise ValueError(f"window {window} s is too small for fs={f} (need at least 2 samples)")

    seg = e[i0:]
    # E[k] = sum of energy from sample i0+k to the end (Schroeder integral).
    E = e[i0:].sum() - np.cumsum(seg) + seg
    n_win = i1 - i0 + 1
    logE = np.log10(E[:n_win])
    if not np.all(np.isfinite(logE)):
        warnings.warn(
            "rt60_schroeder: Schroeder integral hit 0 in the window — no decay fit",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf")

    x = np.arange(n_win, dtype=np.float64) / f
    a, _b = np.polyfit(x, logE, 1)  # slope a of log10(E) = a·t + b  (dB/s)
    if not np.isfinite(a) or a >= 0.0:
        warnings.warn(
            "rt60_schroeder: non-negative decay slope — returning inf (no decay)",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf")
    # a is the log10(E) slope per second → 10·|a| dB/s; RT60 = 60 dB / decay rate.
    return float(6.0 / abs(a))


# --------------------------------------------------------------------------- #
# Delay spread
# --------------------------------------------------------------------------- #
def tau_mean(rir: np.ndarray, fs: float) -> float:
    """Mean delay spread of the RIR's energy-weighted delay distribution.

    ``τ_mean = Σ t_i·e_i / Σ e_i`` where ``e_i = |h_i|²`` and ``t_i`` is the
    time of the ``i``-th sample (since the first sample). A delta impulse
    gives ``0.0``.
    """
    _h, _f, t, e = _validate_rir_and_fs(rir, fs)
    total = float(e.sum())
    if total == 0.0:
        return 0.0
    return float(np.dot(t, e) / total)


def tau_rms(rir: np.ndarray, fs: float) -> float:
    """RMS delay spread of the RIR's energy-weighted delay distribution.

    ``τ_rms = sqrt( Σ t_i²·e_i / Σ e_i − τ_mean² )``. A delta impulse gives
    ``0.0``. This is the quantity the PHY's coherence bandwidth
    ``B_c ≈ 1 / (5 τ_rms)`` and the ``τ_rms ≈ 0.205·RT60`` diffuse-field check
    (PLAN.md §2.2, S11.3) are built on.
    """
    _h, _f, t, e = _validate_rir_and_fs(rir, fs)
    total = float(e.sum())
    if total == 0.0:
        return 0.0
    mean = float(np.dot(t, e) / total)
    second = float(np.dot(t * t, e) / total)
    return float(np.sqrt(max(0.0, second - mean * mean)))


# --------------------------------------------------------------------------- #
# Early/late energy fraction times (B50 / B90)
# --------------------------------------------------------------------------- #
def _fraction_time(rir: np.ndarray, fs: float, fraction: float) -> float:
    """Time at which the cumulative RIR energy reaches ``fraction`` of the total.

    Uses ``np.searchsorted`` on the cumulative energy and linear interpolation
    between the two bracketing samples, so exponential / linear / flat RIRs
    recover their analytic fraction times to within the discretisation.
    A single-sample RIR returns ``0.0``.
    """
    _h, _f, t, e = _validate_rir_and_fs(rir, fs)
    frac = float(fraction)
    if not (0.0 < frac < 1.0):
        raise ValueError(f"fraction must be in (0, 1) (got {fraction})")
    total = float(e.sum())
    if total == 0.0:
        return 0.0
    if t.size == 1:
        return 0.0
    cum = np.cumsum(e)
    target = frac * total
    idx = int(np.searchsorted(cum, target, side="left"))
    if idx == 0:
        return float(t[0])
    if idx >= t.size:
        return float(t[-1])
    e0, e1 = float(cum[idx - 1]), float(cum[idx])
    if e1 <= e0 or e0 <= 0.0:
        # Flat step, or the fraction is reached in the first non-zero bin
        # (zero-padded impulse): no energy before it, so no interpolation.
        return float(t[idx])
    return float(t[idx - 1] + (t[idx] - t[idx - 1]) * (target - e0) / (e1 - e0))


def b50(rir: np.ndarray, fs: float) -> float:
    """Time (s) at which 50% of the RIR energy has arrived (``B50``).

    For an exponential RIR ``r(t)=e^(−t/λ)``, ``E(t) ∝ e^(−2t/λ)`` so
    ``B50 = (λ/2)·ln(2) ≈ 0.347·λ``; for a linear-energy RIR of duration ``D``
    (``e(t) ∝ 1 − t/D``), ``B50 = D·(1 − 1/√2) ≈ 0.293·D``; for a flat RIR of
    duration ``D``, ``D/2``.
    """
    return _fraction_time(rir, fs, 0.5)


def b90(rir: np.ndarray, fs: float) -> float:
    """Time (s) at which 90% of the RIR energy has arrived (``B90``).

    For an exponential RIR ``r(t)=e^(−t/λ)``, ``B90 = (λ/2)·ln(10) ≈ 1.151·λ``;
    for a linear-energy RIR of duration ``D``, ``B90 = D·(1 − √0.1) ≈ 0.684·D``;
    for a flat RIR of duration ``D``, ``0.9·D``.
    """
    return _fraction_time(rir, fs, 0.9)


# --------------------------------------------------------------------------- #
# Demo (``python -m auditory_handshake.metrics.channel_metrics``)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    """Print RT60 / τ / B on a synthetic RIR, and (if available) a real RIR."""
    fs = 16000
    t = np.arange(int(0.2 * fs)) / fs

    print("S10.1 — channel metrics on a Room Impulse Response")
    print("=" * 60)
    print("Synthetic RIR: r(t) = exp(−t/λ), λ = 0.050 s")
    print(f"  fs = {fs} Hz, N = {t.size}")
    h = np.exp(-t / 0.050)
    rt60 = rt60_schroeder(h, fs)
    tm, tr = tau_mean(h, fs), tau_rms(h, fs)
    print(
        f"  RT60 (Schroeder) : {rt60 * 1000:8.2f} ms   (exact 6.908λ = {0.050 * 6.908 * 1000:6.2f})"
    )
    print(f"  τ_mean           : {tm * 1000:8.2f} ms   (exact λ/2    = {0.050 * 0.5 * 1000:6.2f})")
    print(f"  τ_rms            : {tr * 1000:8.2f} ms   (exact λ/2    = {0.050 * 0.5 * 1000:6.2f})")
    print(
        f"  B50              : {b50(h, fs) * 1000:8.2f} ms   (exact 0.347λ = {0.050 * 0.347 * 1000:6.2f})"
    )
    print(
        f"  B90              : {b90(h, fs) * 1000:8.2f} ms   (exact 1.151λ = {0.050 * 1.151 * 1000:6.2f})"
    )
    print(f"  0.205·RT60 check : {0.205 * rt60 * 1000:8.2f} ms   (diffuse-field lower bound)")

    try:
        from auditory_handshake.acoustics.channel import compute_rir
        from auditory_handshake.config import list_room_presets, load_room_preset

        print()
        print("Real RIR (pyroomacoustics ShoeBox), first preset, design RT60 = 0.8 s:")
        name = list_room_presets()[0]
        preset = load_room_preset(name)
        rir = compute_rir(preset, rt60_seconds=0.8, fs=fs)
        rt60r = rt60_schroeder(rir, fs)
        print(f"  preset = {name!r}")
        print(f"  RT60 (Schroeder) : {rt60r * 1000:8.2f} ms")
        print(f"  τ_mean           : {tau_mean(rir, fs) * 1000:8.2f} ms")
        print(f"  τ_rms            : {tau_rms(rir, fs) * 1000:8.2f} ms")
        print(f"  0.205·RT60 check : {0.205 * rt60r * 1000:8.2f} ms")
    except Exception as exc:  # pragma: no cover — demo only
        print(f"\n  (real-RIR demo skipped: {exc})")


if __name__ == "__main__":
    _demo()
