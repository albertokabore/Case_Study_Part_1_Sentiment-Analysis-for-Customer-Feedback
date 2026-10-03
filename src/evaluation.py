"""Model evaluation: metrics, statistical comparison and diagnostic plots.

Metrics (all on the same held-out test set)
-------------------------------------------
* Accuracy
* Precision, recall and F1 per class, plus their **macro** averages (each class
  weighted equally – the primary criterion given the 63 / 21 / 16 % imbalance)
  and **weighted** averages (by class support).
* One-vs-rest ROC AUC (macro), from probabilities or SVM decision margins.

Statistical comparison
----------------------
* 95 % bootstrap confidence intervals (1,000 resamples) for accuracy and macro-F1.
* McNemar's test (Dietterich, 1998) between the best model and every other model:
  does the pair disagree on test tweets more often in one direction than chance?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from . import config
from .plotting import BLUE_RAMP, MODEL_COLORS, TEXT_PRIMARY, TEXT_SECONDARY, save

SCORE_COLUMNS = [f"score_{label}" for label in config.LABELS]


# --------------------------------------------------------------------------- #
# Prediction persistence – every model writes the same CSV schema
# --------------------------------------------------------------------------- #
def save_predictions(slug: str, tweet_ids, y_true, y_pred, scores, directory: Path = config.PREDICTIONS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"tweet_id": tweet_ids, "y_true": y_true, "y_pred": y_pred})
    df[SCORE_COLUMNS] = np.asarray(scores)
    path = directory / f"{slug}_test.csv"
    df.to_csv(path, index=False)
    return path


def load_predictions(directory: Path = config.PREDICTIONS_DIR) -> dict[str, pd.DataFrame]:
    return {p.stem.removesuffix("_test"): pd.read_csv(p) for p in sorted(directory.glob("*_test.csv"))}


def validate_predictions(predictions: dict[str, pd.DataFrame], expected_test: pd.DataFrame | None = None) -> None:
    """Reject incomplete, inconsistent or stale predictions before comparing models."""
    if not predictions:
        raise ValueError("No saved predictions found.")
    reference = expected_test.set_index("tweet_id")["label"] if expected_test is not None else None
    for slug, frame in predictions.items():
        required = {"tweet_id", "y_true", "y_pred", *SCORE_COLUMNS}
        if not required.issubset(frame.columns) or frame.empty or not frame["tweet_id"].is_unique:
            raise ValueError(f"{slug}: missing columns, empty predictions or duplicate tweet IDs.")
        truth = frame.set_index("tweet_id")["y_true"]
        if reference is None:
            reference = truth
        if set(truth.index) != set(reference.index) or not np.array_equal(truth.values, reference.reindex(truth.index).values):
            raise ValueError(f"{slug}: test IDs or true labels do not match the expected test set.")
        if not frame[["y_true", "y_pred"]].isin(range(len(config.LABELS))).all().all():
            raise ValueError(f"{slug}: invalid class labels.")
        if not np.isfinite(frame[SCORE_COLUMNS].to_numpy()).all():
            raise ValueError(f"{slug}: non-finite decision scores.")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, scores=None) -> dict[str, float]:
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f_mac,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f_w,
    }
    if scores is not None:
        metrics["roc_auc_macro_ovr"] = roc_auc_score(
            label_binarize(y_true, classes=range(len(config.LABELS))), np.asarray(scores), average="macro"
        )
    return {k: float(v) for k, v in metrics.items()}


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(config.LABELS)), zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=config.LABELS)


def bootstrap_ci(y_true, y_pred, n_resamples: int = 1000, alpha: float = 0.05, seed: int = config.SEED) -> dict:
    """Percentile bootstrap CIs for accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    acc, f1 = np.empty(n_resamples), np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        acc[i] = accuracy_score(y_true[idx], y_pred[idx])
        f1[i] = f1_score(y_true[idx], y_pred[idx], average="macro")
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy_ci_low": float(np.percentile(acc, lo)), "accuracy_ci_high": float(np.percentile(acc, hi)),
        "f1_macro_ci_low": float(np.percentile(f1, lo)), "f1_macro_ci_high": float(np.percentile(f1, hi)),
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact/χ² McNemar test on the 2×2 table of (A correct?, B correct?)."""
    a_ok, b_ok = np.asarray(pred_a) == np.asarray(y_true), np.asarray(pred_b) == np.asarray(y_true)
    table = [[int(np.sum(a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))],
             [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))]]
    discordant = table[0][1] + table[1][0]
    result = mcnemar(table, exact=discordant < 25, correction=True)
    return {"a_only_correct": table[0][1], "b_only_correct": table[1][0],
            "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def comparison_table(predictions: dict[str, pd.DataFrame], names: dict[str, str]) -> pd.DataFrame:
    """One row per model: point metrics + bootstrap CIs, sorted by macro-F1."""
    validate_predictions(predictions)
    rows = []
    for slug, df in predictions.items():
        row = {"slug": slug, "model": names.get(slug, slug)}
        row.update(compute_metrics(df["y_true"], df["y_pred"], df[SCORE_COLUMNS].values))
        row.update(bootstrap_ci(df["y_true"], df["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def significance_table(predictions: dict[str, pd.DataFrame], best_slug: str, names: dict[str, str]) -> pd.DataFrame:
    validate_predictions(predictions)
    best = predictions[best_slug]
    rows = []
    for slug, df in predictions.items():
        if slug == best_slug:
            continue
        merged = best.merge(df, on="tweet_id", suffixes=("_best", "_other"))
        res = mcnemar_test(merged["y_true_best"], merged["y_pred_best"], merged["y_pred_other"])
        rows.append({"comparison": f"{names.get(best_slug, best_slug)} vs {names.get(slug, slug)}", **res})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["comparison", "a_only_correct", "b_only_correct", "statistic", "p_value",
                                     "p_value_holm", "significant_at_0.05", "significant_holm_at_0.05"])
    out["significant_at_0.05"] = out["p_value"] < 0.05
    reject, adjusted, _, _ = multipletests(out["p_value"], alpha=0.05, method="holm")
    out["p_value_holm"] = adjusted
    out["significant_holm_at_0.05"] = reject
    return out.sort_values("p_value").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_pred, title: str, ax: plt.Axes | None = None, normalize: bool = True) -> plt.Axes:
    """Row-normalised confusion matrix (recall on the diagonal) with counts."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(config.LABELS)))
    share = cm / cm.sum(axis=1, keepdims=True)
    ax = ax or plt.subplots(figsize=(4.2, 3.8))[1]
    ax.imshow(share if normalize else cm, cmap=BLUE_RAMP, vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if share[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=9, color=colour)
    ax.set_xticks(range(3), config.LABELS)
    ax.set_yticks(range(3), config.LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_confusion_grid(predictions: dict[str, pd.DataFrame], names: dict[str, str], path: Path, order: list[str]) -> Path:
    n = len(order)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, slug in zip(axes, order):
        df = predictions[slug]
        plot_confusion_matrix(df["y_true"], df["y_pred"], names.get(slug, slug), ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on the test set (row-normalised: diagonal = recall)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return save(fig, path)


def plot_roc_per_class(df: pd.DataFrame, title: str, path: Path) -> Path:
    """One-vs-rest ROC curves for each sentiment class of a single model."""
    y_bin = label_binarize(df["y_true"], classes=range(len(config.LABELS)))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for k, label in enumerate(config.LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, k], df[SCORE_COLUMNS[k]])
        ax.plot(fpr, tpr, color=config.LABEL_COLORS[label], label=f"{label} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#b9b8b2", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    return save(fig, path)


def plot_roc_by_model(predictions: dict[str, pd.DataFrame], names: dict[str, str], path: Path, order: list[str]) -> Path:
    """Macro-averaged one-vs-rest ROC curve for each model."""
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for slug in order:
        df = predictions[slug]
        y_bin = label_binarize(df["y_true"], classes=range(len(config.LABELS)))
        tprs = []
        for k in range(len(config.LABELS)):
            fpr, tpr, _ = roc_curve(y_bin[:, k], df[SCORE_COLUMNS[k]])
            tprs.append(np.interp(grid, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        ax.plot(grid, mean_tpr, color=MODEL_COLORS.get(slug), label=f"{names.get(slug, slug)} (AUC = {auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#b9b8b2")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (macro average)")
    ax.set_title("Macro-averaged one-vs-rest ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    return save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars of macro-F1 with 95 % bootstrap CIs, plus accuracy markers."""
    t = table.sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(t) + 1.4))
    y = np.arange(len(t))
    colours = [MODEL_COLORS.get(s, "#2a78d6") for s in t["slug"]]
    err = np.vstack([t["f1_macro"] - t["f1_macro_ci_low"], t["f1_macro_ci_high"] - t["f1_macro"]])
    ax.barh(y, t["f1_macro"], color=colours, height=0.6, xerr=err,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 3})
    ax.scatter(t["accuracy"], y, marker="D", s=36, color="white", edgecolor=TEXT_PRIMARY, zorder=3, label="Accuracy")
    for yi, (f1, acc) in enumerate(zip(t["f1_macro"], t["accuracy"])):
        ax.text(max(f1, acc) + 0.025, yi, f"F1 {f1:.3f} · Acc {acc:.3f}", va="center", fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_yticks(y, t["model"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Score on held-out test set")
    ax.set_title("Model comparison – macro-F1 (bars, 95 % CI) and accuracy (◆)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return save(fig, path)


def plot_per_class_f1(predictions: dict[str, pd.DataFrame], names: dict[str, str], path: Path, order: list[str]) -> Path:
    """Grouped bars: F1 per sentiment class for each model."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.8 / len(order)
    x = np.arange(len(config.LABELS))
    for i, slug in enumerate(order):
        df = predictions[slug]
        f1 = per_class_report(df["y_true"], df["y_pred"])["f1"].values
        ax.bar(x + i * width - 0.4 + width / 2, f1, width=width * 0.92, color=MODEL_COLORS.get(slug),
               label=names.get(slug, slug))
    ax.set_xticks(x, [label.capitalize() for label in config.LABELS])
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.set_title("F1-score by sentiment class")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return save(fig, path)


def plot_training_history(history: list[dict], title: str, path: Path) -> Path:
    h = pd.DataFrame(history)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(h["epoch"], h["train_loss"], marker="o", ms=5, color="#2a78d6", label="train")
    a1.plot(h["epoch"], h["val_loss"], marker="o", ms=5, color="#eb6834", label="validation")
    a1.set_title("Loss")
    a1.set_xlabel("Epoch")
    a1.legend()
    a2.plot(h["epoch"], h["val_macro_f1"], marker="o", ms=5, color="#1baf7a")
    best = h["val_macro_f1"].idxmax()
    a2.annotate(f"best {h.loc[best, 'val_macro_f1']:.3f}", (h.loc[best, "epoch"], h.loc[best, "val_macro_f1"]),
                textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8.5)
    a2.set_title("Validation macro-F1")
    a2.set_xlabel("Epoch")
    for a in (a1, a2):
        a.set_xticks(h["epoch"])
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return save(fig, path)
