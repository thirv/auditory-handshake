"""Experiment run directories — the reproducibility contract (PLAN.md §5, §12).

Every experiment run lands in ``results/<experiment>/<UTC-timestamp>/`` and
holds:

- ``config.json``  — the full run config plus seed, git SHA and UTC timestamp
- ``metrics.json`` — the metrics, stamped with seed and git SHA (required by
  §12: "seeds and git SHA recorded in every ``metrics.json``")
- ``figs/``        — every figure, regenerable by a single ``make`` target
- ``tables/``      — CSV/MD tables

Nothing in here touches the acoustic engine; this is pure layout + JSON I/O,
so the whole thing is testable in milliseconds (``make test`` stays sub-second).
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from auditory_handshake import __version__

if TYPE_CHECKING:
    import numpy as np

RESULTS_ROOT = "results"

#: Injectable clock (callable → datetime) for deterministic tests.
type Clock = Callable[[], datetime]

_CONFIG_FILE = "config.json"
_METRICS_FILE = "metrics.json"
_FIGS_DIR = "figs"
_TABLES_DIR = "tables"


def detect_git_sha(cwd: str | Path | None = None) -> str | None:
    """Return the short git SHA of *cwd*'s working tree, or ``None`` if the
    directory is not a git checkout / git is unavailable.

    Never raises: reproducibility stamping must not break a simulation run.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _utc_now(clock: Clock | None = None) -> datetime:
    """Current time as a *naive UTC* datetime (for timestamps in filenames).

    ``clock`` may be injected for deterministic tests; it must be callable
    and return a datetime (aware or naive — both normalise to naive UTC here).
    """
    dt = clock() if clock is not None else datetime.now(UTC)
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


@dataclass(frozen=True)
class RunDir:
    """One experiment run: ``root / experiment / <UTC-timestamp>``.

    The layout is fixed by PLAN.md §5; the helpers below are the *only* place
    the file/dir names are defined, so simulation code never hand-rolls paths.
    """

    experiment: str
    stamp: str
    root: Path

    @property
    def dir(self) -> Path:
        return self.root / self.experiment / self.stamp

    @property
    def config_path(self) -> Path:
        return self.dir / _CONFIG_FILE

    @property
    def metrics_path(self) -> Path:
        return self.dir / _METRICS_FILE

    @property
    def figs_dir(self) -> Path:
        return self.dir / _FIGS_DIR

    @property
    def tables_dir(self) -> Path:
        return self.dir / _TABLES_DIR

    def ensure_dirs(self) -> None:
        """Create the standard sub-directories (``figs/``, ``tables/``)."""
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)


def create_run(
    experiment: str,
    *,
    seed: int,
    config: dict[str, Any] | None = None,
    git_sha: str | None = None,  # ``None`` → auto-detect (``"unknown"`` if no checkout)
    clock: Clock | None = None,
    root: str | Path = RESULTS_ROOT,
) -> RunDir:
    """Create ``results/<experiment>/<UTC-timestamp>/`` and write ``config.json``.

    ``config.json`` captures everything needed to regenerate the run: the
    experiment name, the full config mapping, the recorded seed, the git SHA
    (auto-detected when not passed; ``"unknown"`` outside a checkout), the UTC
    timestamp, and the software versions (PLAN.md §5, §12).
    """
    stamp = _utc_now(clock).strftime("%Y%m%dT%H%M%S")
    run = RunDir(experiment=experiment, stamp=stamp, root=Path(root))
    run.dir.mkdir(parents=True, exist_ok=True)
    run.ensure_dirs()

    sha: str
    if git_sha is None:
        detected = detect_git_sha()
        sha = detected if detected is not None else "unknown"
    else:
        sha = git_sha

    payload: dict[str, Any] = {
        "experiment": experiment,
        "timestamp_utc": stamp,
        "seed": seed,
        "git_sha": sha,
        "config": config or {},
        "versions": {
            "auditory_handshake": __version__,
            "python": platform.python_version(),
        },
    }
    run.config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return run


def generate_audio(
    path: str | Path,
    data: np.ndarray,
    sample_rate: int,
    *,
    normalize_peak: float | None = 0.9,
) -> Path:
    """Write float waveform samples to *path* as a 16-bit PCM WAV (soundfile).

    *data* is 1-D (mono) or 2-D (columns = channels), finite values in
    roughly the ``[-1, 1]`` range. Unless ``normalize_peak`` is ``None``, the
    samples are peak-normalised so ``max|sample| * normalize_peak`` lands at
    the normalised peak — the WAV always stays safely below full scale no
    matter how the source signal was scaled.

    Returns the resolved output path.
    """
    import numpy as np  # local import: keep the audio path out of the fast import

    arr = np.asarray(data)
    if arr.ndim not in (1, 2):
        raise ValueError(f"data must be 1-D (mono) or 2-D (channels), got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contains NaN/Inf — refusing to write a corrupt WAV")
    if normalize_peak is not None:
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 0.0:
            arr = arr * (float(normalize_peak) / peak)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(out, arr, int(sample_rate), subtype="PCM_16")
    return out


def write_metrics(
    run: RunDir,
    metrics: dict[str, Any],
    *,
    seed: int | None = None,
    git_sha: str | None = None,
) -> Path:
    """Write ``metrics.json`` for *run*, stamping in seed and git SHA (PLAN.md §12).

    If ``seed``/``git_sha`` are not given they are read back from the run's
    ``config.json`` so the stamp is guaranteed consistent with the run record.
    Returns the written path.
    """
    recorded: dict[str, Any] = {}
    if run.config_path.exists():
        recorded = json.loads(run.config_path.read_text())

    payload: dict[str, Any] = dict(metrics)
    payload.setdefault("seed", recorded.get("seed"))
    payload.setdefault("git_sha", recorded.get("git_sha"))
    if seed is not None:
        payload["seed"] = seed
    if git_sha is not None:
        payload["git_sha"] = git_sha
    payload.setdefault("experiment", run.experiment)
    payload.setdefault("timestamp_utc", run.stamp)

    path = run.metrics_path
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
