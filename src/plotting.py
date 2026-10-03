"""Shared matplotlib style so every figure in the paper looks like one system.

* Sentiment classes use a diverging palette (``config.LABEL_COLORS``).
* Models use a fixed categorical order (CVD-validated): a model keeps its colour
  in every figure, regardless of its rank.
* Magnitudes (confusion matrices, heatmaps) use a single blue ramp.
* Thin marks, recessive grid, no top/right spines, values labelled directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Fixed model → colour mapping (colour follows the entity, never its rank).
MODEL_ORDER = ["nb_tfidf", "lr_tfidf", "svm_tfidf", "svm_word_char", "lr_word2vec", "bilstm", "distilroberta"]
MODEL_COLORS = dict(zip(MODEL_ORDER, CATEGORICAL))

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#b9b8b2",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


apply_style()
