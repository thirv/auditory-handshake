# pyroomacoustics 0.10.1 API Documentation

## Overview
This document records the verified API surface of pyroomacoustics 0.10.1 as
installed in this environment. Every claim below is a probe result, not a
recollection: the raw evidence is the probe log (a locally-generated artifact,
produced by the command below — not committed), which is the output of

```
python tools/probe_pra_api.py > docs/probe_pra_api.log 2>&1
```

The API differs significantly from the 0.6.x lineage that PLAN.md was drafted
against, so S9-S12 must be written against this document, not against 0.6.x
muscle memory.

## Version & Platform
- Python: 3.12.14
- pyroomacoustics: 0.10.1
- NumPy: 2.5.3
- Platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43

## Module Structure
### Top-level names
The full name list is probe section 1 of `docs/probe_pra_api.log`. The parts that
matter for this project:

| Need | Names |
|---|---|
| Rooms | `Room`, `ShoeBox`, `AnechoicRoom`, `Wall`, `Wall2D` |
| Materials | `Material`, `make_materials`, `materials_absorption_table`, `materials_scattering_table` |
| Geometry | `find_non_convex_walls`, `wall_factory`, `unit_vec2D` |
| RT60 estimators | `measure_rt60`, `rt60_sabine`, `rt60_eyring`, `inverse_sabine` |
| Physics / air | `calculate_speed_of_sound`, `air_absorption_table`, `Physics`, `Constants` |
| Submodules | `signal`, `experimental`, `acoustics`, `beamforming`, `doa`, `bss`, `metrics`, `room`, `soundsource`, `directivities`, `datasets`, `libroom` |

### pra.signal module
Module containing signal processing functions and classes for acoustic signal analysis.

### Materials support
- `pra.Material` class exists with constructor: `(self, energy_absorption, scattering=None)`
- `pra.make_materials` function exists
- `pra.Material` uses `.absorption_coeffs` (the 0.6.x name `.absorption` does **not**
  exist: the probe checks it explicitly and gets `False`) and supports flat or
  frequency-dependent absorption
- `pra.Material` public attrs/methods: `absorption_coeffs`, `scattering_coeffs`,
  `all_flat`, `is_freq_flat`, `resample`
- `Material(0.5)` internal state:
  `{'energy_absorption': {'coeffs': [0.5]}, 'scattering': {'coeffs': [0.0]}}`
- Database names are accepted: `Material('anechoic').absorption_coeffs == [1.0]`.
  `materials_absorption_table` has 91 keys (e.g. `acoustical_plaster_25mm`,
  `audience_1_m2`, `audience_floor`) - this is the source for the S9 room presets.

## Core Classes

### Room class
- Constructor signature: `(self, walls, fs=8000, t0=0.0, max_order=1, sigma2_awgn=None, sources=None, mics=None, temperature=None, humidity=None, air_absorption=False, ray_tracing=False, use_rand_ism=False, max_rand_disp=0.08, min_phase=False)`
- Public methods:
  - `add`, `add_microphone`, `add_microphone_array`, `add_soundsource`, `add_source`, `compute_rir`, `direct_snr`, `extrude`, `from_corners`, `get_bbox`, `get_volume`, `get_wall_by_name`, `image_source_model`, `is_inside`, `measure_rt60`, `plot`, `plot_rir`, `ray_tracing`, `rt60_theory`, `set_air_absorption`, `set_ray_tracing`, `set_sound_speed`, `simulate`, `unset_air_absorption`, `unset_ray_tracing`, `wall_area`
- Key methods:
  - `Room.compute_rir` signature: `(self)`
  - `Room.simulate` signature: `(self, snr=None, reference_mic=0, callback_mix=None, callback_mix_kwargs={}, return_premix=False, recompute_rir=False)`
  - `Room.add_source` signature: `(self, position, signal=None, delay=0, directivity=None)`
  - `Room.add_microphone` signature: `(self, loc, fs=None, directivity=None)`
  - `Room.measure_rt60` signature: `(self, decay_db=60, plot=False, label=None, linear_domain_fit=False)`
  - `Room.rt60_theory` signature: `(self, formula='sabine')`
  - `Room.set_air_absorption` signature: `(self, coefficients=None, center_freqs=None, interp_kind='linear')`
  - `Room.wall_area` signature: `(self, wall)`
  - `Room.from_corners` signature: `(corners, fs=8000, t0=0.0, max_order=1, sigma2_awgn=None, sources=None, mics=None, materials=None, **kwargs)`
  - `Room.extrude` signature: `(self, height, v_vec=None, materials=None)`
  - `Room.get_wall_by_name('west')` resolves the default shoebox wall named `west`
  - Note: `Room.add_point_source` (0.6.x name) does not exist

### ShoeBox class
- Constructor signature: `(self, p, fs=8000, t0=0.0, max_order=1, sigma2_awgn=None, sources=None, mics=None, materials=None, temperature=None, humidity=None, air_absorption=False, ray_tracing=False, use_rand_ism=False, max_rand_disp=0.08, min_phase=False)`
- Default `room.walls` is a `list` of **6** `Wall` objects named
  `['west', 'east', 'south', 'north', 'floor', 'ceiling']`, each with
  `absorption = [0.0]` (fully reflective - see gotcha 3)
- `room.walls[i]` exposes `.absorption` (a list, default `[0.0]`) and a `.scatter()`
  method, but **no** `.scattering` attribute — probe section 8 prints
  `Wall instance exposes .scattering -> False`. Scattering coefficients live on
  `Material.scattering_coeffs`, not on the `Wall` object.
- `ShoeBox` has **no** `set_materials`; pass `materials=` to the constructor or build
  `Room(walls=..., materials=...)`
- `Wall` is the pybind11 class `pyroomacoustics.libroom.Wall`

### AnechoicRoom class
- Constructor signature: `(self, dim=3, fs=8000, t0=0.0, sigma2_awgn=None, sources=None, mics=None, temperature=None, humidity=None, air_absorption=False)`
- The useful "no reverb" control for S11 comparisons

## Functions

### measure_rt60 function
- Signature: `(h, fs=1, decay_db=60, energy_thres=1.0, plot=False, rt60_tgt=None, label=None, linear_domain_fit=False)`
- Located in both top-level and pra.experimental modules
- `rt60_sabine`, `rt60_eyring` and `inverse_sabine` are also top-level functions

### calculate_speed_of_sound
- Signature: `(t, h, p)` (temperature, humidity, pressure); used by `Room` when
  `temperature`/`humidity` are supplied

### find_non_convex_walls
- Signature: `(walls)`; takes 2-D walls and returns the wall set for a non-convex
  polygon - the intended entry point for the corridor preset (S9.2), together with
  `pra.Wall2D`

### experimental module
- Module exists with functions:
  - `measure_rt60`
  - `rt60`
- Contains: DelayCalibration, PointCloud, calculate_speed_of_sound, deconvolution, deconvolve, delay_calibration, edm_line_search, exponential_sweep, linear_sweep, localization, measure_ir, physics, point_cloud, rt60, signals, tdoa, tdoa_loc, wiener_deconvolve, window

## Notes / gotchas

1. No `pra.raytrace` function exists in this version. The method `Room.ray_tracing`
   does exist, and `Room.__init__` takes `ray_tracing=`.
2. No `pra.materials` module exists - materials are handled via `pra.Material`,
   `pra.make_materials` and the `materials_*_table` dicts.
3. **Default `ShoeBox` walls have zero absorption (`[0.]`)**: `rt60_theory()` returns
   `inf` and emits a divide-by-zero `RuntimeWarning`. Room presets (S9) must set
   absorption explicitly or every RT60 number downstream is meaningless.
4. **`measure_rt60()` != `rt60_theory()`**: with `max_order` finite the decay is
   truncated, so the measured value saturates. Probe section 9:
   `alpha=0.0 -> theory=inf, measured=0.0902`; `alpha=0.3 -> theory=0.3222,
   measured=0.1035` (RIR length 2044 samples in both cases). S10 must state which
   estimator it uses, and S11 must compare measured vs predicted knowing this bias.
5. No motion/Doppler support found in the installed source (no velocity/Doppler
   references anywhere in the package): S12 must implement moving-source Doppler
   itself (block-wise geometry + per-block resampling).
6. Smoke simulation result: 1 m^3 shoebox, 50 ms tone, fs = 16 kHz.
   `simulate()` returns `None` and records into `mic_array`; `room.rir[0][0]` has
   shape `(317,)`; `mic_array.signals` has shape `(1, 1116)`;
   `room.get_volume() = 1.0`; `room.rt60_theory() = inf`;
   `room.measure_rt60() = array([[0.0098963]])`.
7. `docs/probe_pra_api.log` opens with two `RuntimeWarning`s on stderr before probe
   section 0 (unbuffered stderr is flushed ahead of buffered stdout under
   `> log 2>&1`). They are gotcha 3 firing, not a probe failure - the probe itself
   reports `[OK]` for every check.
