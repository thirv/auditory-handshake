"""Shared plotting setup — one style + one save path for every figure.

Figures are produced from the headless ``Agg`` backend (no display needed;
this matters for CI and remote work). ``apply_style`` enforces the single
project-wide look, and ``save_fig`` writes every figure to ``figs/`` at a
fixed DPI so outputs are reproducible (PLAN.md §12).

The style values are a provisional house default, frozen in the M9 spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.figure import Figure

mpl.use("Agg")

FIGURES_DIR = "figs"


@dataclass(frozen=True)
class FigureStyle:
    """House figure style (single source of truth)."""

    dpi: int = 200
    font_size: float = 10.0
    figure_size: tuple[float, float] = (8.0, 4.5)
    line_width: float = 1.2
    grid_alpha: float = 0.3
    # Palette: colour-blind-safe, print-friendly.
    color_cycle: tuple[str, ...] = (
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
    )


HOUSE_STYLE = FigureStyle()


def apply_style(style: FigureStyle = HOUSE_STYLE) -> None:
    """Apply the house style to matplotlib's global rcParams."""
    mpl.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "font.size": style.font_size,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": style.grid_alpha,
            "grid.linewidth": 0.5,
            "lines.linewidth": style.line_width,
            "axes.prop_cycle": cycler(color=style.color_cycle),
        }
    )


def new_figure(style: FigureStyle = HOUSE_STYLE) -> Figure:
    """Return a new, style-applied figure with a single ready-to-plot Axes."""
    apply_style(style)
    fig, _ = plt.subplots(figsize=style.figure_size)
    return fig


def save_fig(fig: Figure, name: str, out_dir: str | Path = FIGURES_DIR, **kwargs: Any) -> Path:
    """Save *fig* as ``out_dir/name`` (PNG, house DPI). Returns the written path."""
    out = Path(out_dir)
    if not name.lower().endswith(".png"):
        name = name.rsplit(".", 1)[0] + ".png"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight", **kwargs)
    return path
