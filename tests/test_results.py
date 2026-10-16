"""Tests for ``results.py`` — run-directory layout and reproducibility stamping.

All tests run against ``tmp_path`` with an injected clock, so the suite is
deterministic and stays inside the sub-second ``make test`` budget.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from auditory_handshake import __version__
from auditory_handshake.results import (
    RunDir,
    create_run,
    detect_git_sha,
    write_metrics,
)

FIXED_NOW = datetime(2026, 9, 28, 12, 30, 45)
FIXED_STAMP = "20260928T123045"


def _clock() -> datetime:
    return FIXED_NOW


def test_create_run_layout(tmp_path: Path) -> None:
    run = create_run("link_budget", seed=42, clock=_clock, root=tmp_path)
    assert isinstance(run, RunDir)
    # PLAN.md §5: results/<exp>/<UTC-timestamp>/
    assert run.dir == tmp_path / "link_budget" / FIXED_STAMP
    assert run.dir.is_dir()
    assert (run.dir / "figs").is_dir()
    assert (run.dir / "tables").is_dir()


def test_config_json_content(tmp_path: Path) -> None:
    config = {"room": "cabin", "snr_db": 10.0}
    run = create_run(
        "link_budget",
        seed=42,
        config=config,
        git_sha="abc1234",
        clock=_clock,
        root=tmp_path,
    )
    payload = json.loads(run.config_path.read_text())
    assert payload["experiment"] == "link_budget"
    assert payload["seed"] == 42
    assert payload["git_sha"] == "abc1234"
    assert payload["config"] == config
    assert payload["timestamp_utc"] == FIXED_STAMP
    assert payload["versions"]["auditory_handshake"] == __version__


def test_write_metrics_stamps_seed_and_sha(tmp_path: Path) -> None:
    run = create_run("link_budget", seed=7, git_sha="deadbee", clock=_clock, root=tmp_path)
    path = write_metrics(run, {"ber": 0.001})
    assert path == run.dir / "metrics.json"
    payload = json.loads(path.read_text())
    # PLAN.md §12: seeds and git SHA recorded in every metrics.json
    assert payload["ber"] == 0.001
    assert payload["seed"] == 7
    assert payload["git_sha"] == "deadbee"


def test_write_metrics_explicit_override(tmp_path: Path) -> None:
    run = create_run("exp", seed=1, git_sha="aaa111", clock=_clock, root=tmp_path)
    path = write_metrics(run, {"x": 1}, seed=99, git_sha="fff000")
    payload = json.loads(path.read_text())
    assert payload["seed"] == 99
    assert payload["git_sha"] == "fff000"


def test_detect_git_sha_never_raises(tmp_path: Path) -> None:
    # Inside this repo's checkout we expect a real SHA; in a non-repo dir we
    # expect ``None`` — both are valid outcomes, and it must not raise either way.
    inside_repo = detect_git_sha()
    assert inside_repo is None or isinstance(inside_repo, str)
    outside = detect_git_sha(cwd=tmp_path)
    assert outside is None or isinstance(outside, str)
