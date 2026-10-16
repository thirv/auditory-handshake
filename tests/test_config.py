"""Tests for the config loader (S2.4)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from auditory_handshake.config import (
    _BUNDLED_NAMES,
    DevicePreset,
    FrontierConfig,
    Scenario,
    all_room_presets,
    all_scenarios,
    list_devices,
    list_room_presets,
    list_scenarios,
    load_defaults,
    load_device_preset,
    load_frontier,
    load_room_preset,
    load_scenario,
    validate_scenario_references,
)


def test_defaults_has_required_keys() -> None:
    d = load_defaults()
    assert isinstance(d.get("seed"), int)
    assert isinstance(d.get("sample_rate_hz"), int)
    assert d.get("sample_rate_hz", 0) > 0
    bands = d.get("audio_bands_hz")
    assert isinstance(bands, dict)
    for key in ("mfsk", "css"):
        assert key in bands, f"audio_bands_hz missing {key!r}"
        pair = bands[key]
        assert (
            isinstance(pair, list) and len(pair) == 2
        ), f"audio_bands_hz.{key} must be a 2-element list"
        assert pair[0] < pair[1]


def test_bundled_room_presets_all_load() -> None:
    names = list_room_presets()
    for expected in _BUNDLED_NAMES:
        assert expected in names, f"bundled preset {expected!r} missing from {names}"


def test_room_preset_fields() -> None:
    for name in _BUNDLED_NAMES:
        p = load_room_preset(name)
        assert p.name == name
        assert len(p.dimensions_m) == 3
        assert all(d > 0 for d in p.dimensions_m)
        assert len(p.frequencies_hz) >= 2
        for face in ("x", "y", "z"):
            vals = p.absorption[face]
            assert len(vals) == len(p.frequencies_hz)
            assert all(0.0 <= a <= 1.0 for a in vals)


def test_all_room_presets_is_sorted_and_complete() -> None:
    presets = all_room_presets()
    assert [p.name for p in presets] == sorted(p.name for p in presets)
    assert {p.name for p in presets} == set(_BUNDLED_NAMES)


def test_unknown_room_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_room_preset("no_such_room", presets_dir=tmp_path)


def test_bad_yaml_raises(tmp_path: Path) -> None:
    rooms = tmp_path / "rooms"
    rooms.mkdir()
    bad = rooms / "bad.yaml"
    bad.write_text("name: bad\ndimensions_m: [1.0, -2.0, 3.0]\n")
    with pytest.raises(ValueError, match="strictly positive"):
        load_room_preset("bad", presets_dir=tmp_path)


def test_absorption_out_of_range_raises(tmp_path: Path) -> None:
    rooms = tmp_path / "rooms"
    rooms.mkdir()
    bad = rooms / "bad.yaml"
    bad.write_text(
        "name: bad\n"
        "dimensions_m: [1.0, 1.0, 1.0]\n"
        "absorption:\n"
        "  frequencies_hz: [100.0, 200.0]\n"
        "  absorption:\n"
        "    x: [0.5, 1.5]\n"
        "    y: [0.5, 0.5]\n"
        "    z: [0.5, 0.5]\n"
    )
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        load_room_preset("bad", presets_dir=tmp_path)


def test_round_trip_to_dict() -> None:
    from auditory_handshake.config import RoomPreset

    p = load_room_preset("cabin")
    p2 = RoomPreset.from_dict(p.to_dict())
    assert p2 == p


# --- Device presets (S2.6) --------------------------------------------------


def test_device_presets_all_load() -> None:
    from auditory_handshake.config import _BUNDLED_DEVICE_NAMES

    names = list_devices()
    for expected in _BUNDLED_DEVICE_NAMES:
        assert expected in names, f"bundled device {expected!r} missing from {names}"


def test_device_preset_fields() -> None:
    from auditory_handshake.config import _BUNDLED_DEVICE_NAMES

    for name in _BUNDLED_DEVICE_NAMES:
        d = load_device_preset(name)
        assert d.name == name
        assert d.kind in ("loudspeaker", "microphone")
        assert len(d.frequency_response_hz) >= 2
        assert all(f > 0 for f in d.frequency_response_hz)
        assert any(
            b > a
            for a, b in zip(d.frequency_response_hz, d.frequency_response_hz[1:], strict=False)
        )
        assert len(d.frequency_response_db) == len(d.frequency_response_hz)
        assert d.usable_band_hz[0] < d.usable_band_hz[1]
        assert d.max_spl_db > 0.0


def test_device_round_trip() -> None:
    d = load_device_preset("small_loudspeaker")
    d2 = DevicePreset.from_dict(d.to_dict())
    assert d2 == d


def test_device_bad_kind_raises(tmp_path: Path) -> None:
    devices = tmp_path / "devices"
    devices.mkdir()
    (devices / "bad.yaml").write_text(
        "name: bad\n"
        "kind: laser\n"
        "frequency_response_hz: [100.0, 200.0]\n"
        "frequency_response_db: [0.0, 0.0]\n"
        "usable_band_hz: [100.0, 200.0]\n"
        "max_spl_db: 100.0\n"
    )
    with pytest.raises(ValueError, match="kind"):
        load_device_preset("bad", presets_dir=devices)


def test_device_misaligned_response_raises(tmp_path: Path) -> None:
    devices = tmp_path / "devices"
    devices.mkdir()
    (devices / "bad.yaml").write_text(
        "name: bad\n"
        "kind: microphone\n"
        "frequency_response_hz: [100.0, 200.0, 300.0]\n"
        "frequency_response_db: [0.0, 0.0]\n"
        "usable_band_hz: [100.0, 200.0]\n"
        "max_spl_db: 100.0\n"
    )
    with pytest.raises(ValueError, match="align"):
        load_device_preset("bad", presets_dir=devices)


# --- Scenarios (S2.7) -------------------------------------------------------


def test_scenarios_all_load() -> None:
    from auditory_handshake.config import _BUNDLED_SCENARIO_NAMES

    names = list_scenarios()
    for expected in _BUNDLED_SCENARIO_NAMES:
        assert expected in names, f"bundled scenario {expected!r} missing from {names}"


def test_scenario_fields() -> None:
    from auditory_handshake.config import _BUNDLED_SCENARIO_NAMES

    for name in _BUNDLED_SCENARIO_NAMES:
        s = load_scenario(name)
        assert s.name == name
        assert s.room
        assert s.tx_device
        assert s.rx_device
        assert s.frame_mode in ("shout", "one_half_way", "two_way", "three_way")
        assert s.drone_speed_ms >= 0.0
        assert isinstance(s.seed, int)
        for n in s.noise_sources:
            assert n.kind
            assert n.level_db_spl > 0.0
            assert n.distance_m > 0.0


def test_scenario_round_trip() -> None:
    s = load_scenario("drone_arrival")
    s2 = Scenario.from_dict(s.to_dict())
    assert s2 == s


def test_scenario_bad_frame_mode_raises(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "bad.yaml").write_text(
        "name: bad\n"
        "room: office\n"
        "tx_device: small_loudspeaker\n"
        "rx_device: mems_microphone\n"
        "noise_sources: []\n"
        "drone_speed_ms: 0.0\n"
        "frame_mode: full_duplex\n"
        "seed: 42\n"
    )
    with pytest.raises(ValueError, match="frame_mode"):
        load_scenario("bad", presets_dir=scenarios)


def test_all_scenarios_is_sorted_and_complete() -> None:
    from auditory_handshake.config import _BUNDLED_SCENARIO_NAMES

    presets = all_scenarios()
    assert [p.name for p in presets] == sorted(p.name for p in presets)
    assert {p.name for p in presets} == set(_BUNDLED_SCENARIO_NAMES)


# --- Cross-reference validation (S2.6) --------------------------------------


def test_scenario_refs_resolve() -> None:
    problems = validate_scenario_references()
    assert problems == [], problems


def test_validate_scenario_references_reports_missing(tmp_path: Path) -> None:
    base = tmp_path / "configs"
    rooms = base / "rooms"
    devices = base / "devices"
    scenarios = base / "scenarios"
    for d in (rooms, devices, scenarios):
        d.mkdir(parents=True)
    (rooms / "small_office.yaml").write_text(
        "name: small_office\n"
        "dimensions_m: [5.0, 4.0, 2.5]\n"
        "absorption:\n"
        "  frequencies_hz: [125.0, 250.0]\n"
        "  absorption:\n"
        "    x: [0.1, 0.2]\n"
        "    y: [0.1, 0.2]\n"
        "    z: [0.1, 0.2]\n"
    )
    (devices / "small_loudspeaker.yaml").write_text(
        "name: small_loudspeaker\n"
        "kind: loudspeaker\n"
        "frequency_response_hz: [100.0, 200.0]\n"
        "frequency_response_db: [-6.0, 0.0]\n"
        "usable_band_hz: [100.0, 200.0]\n"
        "max_spl_db: 95.0\n"
    )
    (devices / "mic.yaml").write_text(
        "name: mic\n"
        "kind: microphone\n"
        "frequency_response_hz: [100.0, 200.0]\n"
        "frequency_response_db: [0.0, 0.0]\n"
        "usable_band_hz: [100.0, 200.0]\n"
        "max_spl_db: 120.0\n"
    )
    (scenarios / "broken.yaml").write_text(
        "name: broken\n"
        "room: no_such_room\n"
        "tx_device: no_such_tx\n"
        "rx_device: mic\n"
        "noise_sources: []\n"
        "drone_speed_ms: 0.0\n"
        "frame_mode: shout\n"
        "seed: 0\n"
    )
    problems = validate_scenario_references(base)
    assert len(problems) == 2, problems
    assert any("room" in p and "no_such_room" in p for p in problems)
    assert any("tx_device" in p and "no_such_tx" in p for p in problems)
    assert not any("rx_device" in p for p in problems)

    # Fix the references -> clean.
    (scenarios / "broken.yaml").write_text(
        "name: broken\n"
        "room: small_office\n"
        "tx_device: small_loudspeaker\n"
        "rx_device: mic\n"
        "noise_sources: []\n"
        "drone_speed_ms: 0.0\n"
        "frame_mode: shout\n"
        "seed: 0\n"
    )
    assert validate_scenario_references(base) == []


def test_validate_scenario_references_tolerates_missing_dirs(tmp_path: Path) -> None:
    # Only a scenarios/ dir; rooms/ and devices/ absent -> no problems reported
    # (a stripped checkout must not fail validation, see the helper docstring).
    base = tmp_path / "configs"
    scenarios = base / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "solo.yaml").write_text(
        "name: solo\n"
        "room: some_room\n"
        "tx_device: some_tx\n"
        "rx_device: some_rx\n"
        "noise_sources: []\n"
        "drone_speed_ms: 0.0\n"
        "frame_mode: shout\n"
        "seed: 0\n"
    )
    assert validate_scenario_references(base) == []


# --- C2 frontier sweep (S17.2) ---------------------------------------------


def test_load_frontier_bundled() -> None:
    fc = load_frontier()
    assert isinstance(fc, FrontierConfig)
    # Band candidates are ordered, strictly increasing pairs.
    assert len(fc.band_candidates) >= 1
    for lo, hi in fc.band_candidates:
        assert 0.0 < lo < hi
    # Budgets / axes are sensible.
    assert fc.spl_db > 0.0
    assert math.isfinite(fc.snr_db)
    assert 0.0 < fc.bler_max <= 1.0
    assert fc.ttch_max_s > 0.0
    assert fc.symbols_per_second > 0.0
    assert fc.n_chirps >= 1
    assert all(v > 0.0 for v in fc.lambda_grid)
    assert fc.aggregator in ("best", "mean", "consensus")
    assert fc.room


def test_frontier_round_trip_to_dict() -> None:
    fc = load_frontier()
    d = fc.to_dict()
    fc2 = FrontierConfig.from_dict(d)
    assert fc2 == fc


def _frontier_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "band_candidates": [[300.0, 1200.0]],
        "spl_db": 80.0,
        "snr_db": 20.0,
        "bler_max": 0.1,
        "ttch_max_s": 0.5,
        "symbols_per_second": 64.0,
        "n_chirps": 3,
        "lambda_grid": [0.01, 0.1, 1.0],
        "aggregator": "best",
        "room": "small_office",
    }
    base.update(overrides)
    return base


def test_frontier_from_dict_round_trip() -> None:
    fc = FrontierConfig.from_dict(_frontier_dict())
    assert fc == FrontierConfig.from_dict(fc.to_dict())


def test_frontier_use_perceptual_cost_default_and_round_trip() -> None:
    # Opt-in toggle defaults to the legacy parametric dB cost (False).
    assert FrontierConfig.from_dict(_frontier_dict()).use_perceptual_cost is False

    # ...and is honoured when set and preserved across a round-trip.
    enabled = FrontierConfig.from_dict(_frontier_dict(use_perceptual_cost=True))
    assert enabled.use_perceptual_cost is True
    assert "use_perceptual_cost" in enabled.to_dict()
    assert enabled.to_dict()["use_perceptual_cost"] is True
    assert FrontierConfig.from_dict(enabled.to_dict()) == enabled


def test_frontier_perceptual_signal_and_noise_defaults() -> None:
    # Defaults: the reference-ramp "noise" type with "broadband" loudness reference.
    fc = FrontierConfig.from_dict(_frontier_dict())
    assert fc.perceptual_signal_type == "noise"
    assert fc.perceptual_noise_kind == "broadband"


def test_frontier_perceptual_signal_and_noise_round_trip() -> None:
    # Modulated-carrier types are accepted and preserved across a round-trip.
    for signal_type in ("noise", "sharp", "rough", "css", "mfsk"):
        fc = FrontierConfig.from_dict(
            _frontier_dict(perceptual_signal_type=signal_type, perceptual_noise_kind="pink")
        )
        assert fc.perceptual_signal_type == signal_type
        assert fc.perceptual_noise_kind == "pink"
        d = fc.to_dict()
        assert d["perceptual_signal_type"] == signal_type
        assert d["perceptual_noise_kind"] == "pink"
        assert FrontierConfig.from_dict(d) == fc


def test_frontier_invalid_perceptual_signal_type_raises() -> None:
    for bad in ("qam", "", 12, None):
        with pytest.raises(ValueError, match="perceptual_signal_type"):
            FrontierConfig.from_dict(_frontier_dict(perceptual_signal_type=bad))


def test_frontier_invalid_perceptual_noise_kind_raises() -> None:
    for bad in ("white", "", 3, None):
        with pytest.raises(ValueError, match="perceptual_noise_kind"):
            FrontierConfig.from_dict(_frontier_dict(perceptual_noise_kind=bad))


def test_frontier_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_frontier(presets_dir=tmp_path)


def test_frontier_band_lo_ge_hi_raises() -> None:
    for bad in ([[1200.0, 300.0]], [[300.0, 300.0]]):
        with pytest.raises(ValueError, match="must be < high frequency"):
            FrontierConfig.from_dict(_frontier_dict(band_candidates=bad))


def test_frontier_band_not_pair_raises() -> None:
    with pytest.raises(ValueError, match="2-element"):
        FrontierConfig.from_dict(_frontier_dict(band_candidates=[[300.0, 1200.0, 4000.0]]))


def test_frontier_bands_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        FrontierConfig.from_dict(_frontier_dict(band_candidates=[]))


def test_frontier_bler_max_out_of_range_raises() -> None:
    for bad in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError, match="bler_max"):
            FrontierConfig.from_dict(_frontier_dict(bler_max=bad))


def test_frontier_bler_max_bool_raises() -> None:
    with pytest.raises(ValueError, match="expected a number"):
        FrontierConfig.from_dict(_frontier_dict(bler_max=True))


def test_frontier_n_chirps_invalid_raises() -> None:
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="n_chirps"):
            FrontierConfig.from_dict(_frontier_dict(n_chirps=bad))


def test_frontier_n_chirps_bool_raises() -> None:
    with pytest.raises(ValueError, match="expected an int"):
        FrontierConfig.from_dict(_frontier_dict(n_chirps=True))


def test_frontier_bad_aggregator_raises() -> None:
    with pytest.raises(ValueError, match="invalid aggregator"):
        FrontierConfig.from_dict(_frontier_dict(aggregator="median"))


def test_frontier_missing_aggregator_raises() -> None:
    d = _frontier_dict()
    del d["aggregator"]
    with pytest.raises(ValueError, match="aggregator"):
        FrontierConfig.from_dict(d)


def test_frontier_missing_room_raises() -> None:
    d = _frontier_dict()
    del d["room"]
    with pytest.raises(ValueError, match="room"):
        FrontierConfig.from_dict(d)


def test_frontier_negative_spl_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        FrontierConfig.from_dict(_frontier_dict(spl_db=-1.0))


def test_frontier_lambda_grid_non_positive_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        FrontierConfig.from_dict(_frontier_dict(lambda_grid=[0.01, 0.0, 1.0]))


def test_frontier_from_dict_non_mapping_raises() -> None:
    with pytest.raises(ValueError, match="expected a mapping"):
        FrontierConfig.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]
