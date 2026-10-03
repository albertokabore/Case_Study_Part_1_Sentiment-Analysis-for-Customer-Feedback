"""Central configuration: file locations, random seed and experiment constants.

Every script and the notebook import from here so that a single change (e.g. the
seed or the test-set fraction) propagates consistently through the whole project.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Tweets.csv"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = ROOT_DIR / "models"  # trained weights; git-ignored (large binaries)

# --------------------------------------------------------------------------- #
# Experiment constants
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.15  # held-out test set, touched once for final evaluation
VAL_SIZE = 0.15  # validation set, used for neural early stopping only
CV_FOLDS = 5  # stratified k-fold for classical hyper-parameter search

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Sentiment is a polarity, so it uses a diverging scheme: red pole, grey
# neutral midpoint, blue pole (CVD-validated; red/blue ΔE 21.6 under protanopia).
LABEL_COLORS = {"negative": "#e34948", "neutral": "#a3a29c", "positive": "#2a78d6"}

# In the published dataset the "Delta" airline label is attached to tweets that
# are addressed to @JetBlue (≈90 % of that group). We correct the label so that
# airline-level business insights are not misattributed. See data_loading.py.
AIRLINE_LABEL_CORRECTIONS = {"Delta": "JetBlue"}


def ensure_dirs() -> None:
    """Create every output directory if it does not yet exist."""
    for path in (RESULTS_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # PyTorch is optional for the classical models
        pass
