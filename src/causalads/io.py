"""Results persistence and a single chart style for every figure."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from causalads import config as C  # noqa: E402

# Validated categorical palette (light mode), fixed order. Use slots in order; charts
# that plot many overlapping points (scatter) should use at most the first three.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")  # negative pole, neutral midpoint, positive pole
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
REFERENCE = "#8a8984"  # zero lines, random-baseline curves


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED, "text.color": TEXT, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 110,
    })


def save_fig(fig: "plt.Figure", name: str) -> Path:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(obj: dict, name: str) -> Path:
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    path = C.RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_default))
    return path


def load_json(name: str) -> dict:
    return json.loads((C.RESULTS / f"{name}.json").read_text())
