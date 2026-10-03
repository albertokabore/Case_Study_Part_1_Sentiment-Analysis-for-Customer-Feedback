"""Exploratory data analysis: summary tables and figures for the paper.

Includes a statistically grounded "distinctive terms" analysis – the weighted
log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008),
which, unlike raw frequency, does not just surface words that are common
everywhere ("flight") and handles rare words without over-ranking them.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data_loading import summary_statistics
from .features import project_feature_space
from .plotting import BLUE_RAMP, TEXT_PRIMARY, TEXT_SECONDARY, save
from .preprocessing import TweetPreprocessor


def log_odds_dirichlet(token_lists: list[list[str]], labels: np.ndarray, top_n: int = 15,
                       min_count: int = 10) -> pd.DataFrame:
    """Weighted log-odds (z-scores) of each word for one class vs. all other classes."""
    total = Counter(t for toks in token_lists for t in toks)
    vocab = [w for w, c in total.items() if c >= min_count]
    prior = np.array([total[w] for w in vocab], dtype=float)
    a0 = prior.sum()
    rows = []
    for k, label in enumerate(config.LABELS):
        in_c = Counter(t for toks, y in zip(token_lists, labels) if y == k for t in toks)
        y_i = np.array([in_c[w] for w in vocab], dtype=float)
        y_j = np.array([total[w] - in_c[w] for w in vocab], dtype=float)
        n_i, n_j = y_i.sum(), y_j.sum()
        delta = (np.log((y_i + prior) / (n_i + a0 - y_i - prior))
                 - np.log((y_j + prior) / (n_j + a0 - y_j - prior)))
        var = 1.0 / (y_i + prior) + 1.0 / (y_j + prior)
        z = delta / np.sqrt(var)
        for idx in np.argsort(-z)[:top_n]:
            rows.append({"sentiment": label, "term": vocab[idx], "z_score": z[idx], "count_in_class": int(y_i[idx])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_sentiment_distribution(df: pd.DataFrame, path: Path) -> Path:
    counts = df["airline_sentiment"].value_counts().reindex(config.LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, color=[config.LABEL_COLORS[c] for c in counts.index], height=0.6)
    for yi, (label, n) in zip(y, counts.items()):
        ax.text(n + 120, yi, f"{n:,}  ({n / counts.sum():.1%})", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_yticks(y, [c.capitalize() for c in counts.index])
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_xlabel("Number of tweets")
    ax.set_title(f"Sentiment distribution (n = {counts.sum():,})")
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def fig_sentiment_by_airline(df: pd.DataFrame, path: Path) -> Path:
    share = pd.crosstab(df["airline"], df["airline_sentiment"], normalize="index")[config.LABELS]
    n = df["airline"].value_counts()
    share = share.sort_values("negative")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for label in config.LABELS:
        vals = share[label].values
        ax.barh(y, vals, left=left, color=config.LABEL_COLORS[label], height=0.62,
                edgecolor="white", linewidth=2, label=label.capitalize())
        for yi, (l0, v) in enumerate(zip(left, vals)):
            if v >= 0.08:
                ax.text(l0 + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if label != "neutral" else TEXT_PRIMARY)
        left += vals
    ax.set_yticks(y, [f"{a}  (n={n[a]:,})" for a in share.index])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Sentiment mix by airline (sorted by share of negative tweets)")
    ax.grid(False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return save(fig, path)


def fig_negative_reasons(df: pd.DataFrame, path: Path) -> Path:
    reasons = df["negativereason"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(reasons.index, reasons.values, color=config.LABEL_COLORS["negative"], height=0.62)
    for yi, v in enumerate(reasons.values):
        ax.text(v + 30, yi, f"{v:,} ({v / reasons.sum():.0%})", va="center", fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_xlim(0, reasons.max() * 1.25)
    ax.set_xlabel("Negative tweets")
    ax.set_title("Why customers complain (annotated reason for negative tweets)")
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def fig_reasons_by_airline(df: pd.DataFrame, path: Path) -> Path:
    neg = df[df["negativereason"].notna()]
    share = pd.crosstab(neg["airline"], neg["negativereason"], normalize="index")
    share = share[share.sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.imshow(share.values, cmap=BLUE_RAMP, aspect="auto", vmin=0, vmax=share.values.max())
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > share.values.max() * 0.55 else TEXT_PRIMARY)
    ax.set_xticks(range(share.shape[1]), share.columns, rotation=30, ha="right")
    ax.set_yticks(range(share.shape[0]), share.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Complaint reasons by airline (share of each airline's negative tweets)")
    return save(fig, path)


def fig_length_by_sentiment(df: pd.DataFrame, path: Path) -> Path:
    words = df["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    data = [words[df["airline_sentiment"] == c] for c in config.LABELS]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": TEXT_PRIMARY, "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], config.LABELS):
        patch.set_facecolor(config.LABEL_COLORS[c])
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
    for i, d in enumerate(data, 1):
        ax.text(d.max() + 0.5, i, f"median {int(d.median())} words", va="center", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_yticks([1, 2, 3], [c.capitalize() for c in config.LABELS])
    ax.set_xlim(0, words.max() + 9)
    ax.set_xlabel("Words per tweet")
    ax.set_title("Tweet length by sentiment – complaints are longer")
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def fig_distinctive_terms(terms: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, label in zip(axes, config.LABELS):
        t = terms[terms["sentiment"] == label].sort_values("z_score")
        ax.barh(t["term"], t["z_score"], color=config.LABEL_COLORS[label], height=0.65)
        ax.set_title(label.capitalize())
        ax.set_xlabel("Log-odds z-score")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle("Most distinctive terms per sentiment (weighted log-odds, informative Dirichlet prior)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return save(fig, path)


def fig_daily_volume(df: pd.DataFrame, path: Path) -> Path:
    daily = (df.assign(day=df["tweet_created"].dt.tz_convert("US/Eastern").dt.date)
             .groupby(["day", "airline_sentiment"], observed=False).size().unstack()[config.LABELS])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for label in config.LABELS:
        ax.plot(daily.index, daily[label], marker="o", ms=4, color=config.LABEL_COLORS[label], label=label.capitalize())
    ax.set_ylabel("Tweets per day")
    ax.set_title("Daily tweet volume by sentiment (Feb 2015, US Eastern time)")
    ax.legend(ncol=3, loc="upper left")
    fig.autofmt_xdate()
    return save(fig, path)


def fig_feature_space(proj: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (cx, cy, title) in zip(axes, [("svd_1", "svd_2", "Truncated SVD (LSA), first two components"),
                                          ("tsne_1", "tsne_2", "t-SNE of 50 SVD components")]):
        # Draw in random order so no class is systematically painted over the others.
        shuffled = proj.sample(frac=1.0, random_state=config.SEED)
        colours = shuffled["label"].map(lambda k: config.LABEL_COLORS[config.LABELS[k]])
        ax.scatter(shuffled[cx], shuffled[cy], s=8, alpha=0.6, c=colours, edgecolor="white", linewidth=0.3)
        for label in config.LABELS:  # legend proxies
            ax.scatter([], [], s=8, color=config.LABEL_COLORS[label], label=label.capitalize())
        ax.set_xlim(*np.percentile(proj[cx], [0.5, 99.5]))
        ax.set_ylim(*np.percentile(proj[cy], [0.5, 99.5]))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1].legend(markerscale=2.5, loc="upper right")
    fig.suptitle(f"TF-IDF feature space (stratified sample, n = {len(proj):,})", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, cleaning_report: dict, tokens: list[list[str]],
            figures_dir: Path, metrics_dir: Path) -> dict[str, pd.DataFrame]:
    """Write every EDA table (CSV) and figure (PNG); return the tables."""
    tables = summary_statistics(df)
    for name, table in tables.items():
        table.to_csv(metrics_dir / f"summary_{name}.csv")

    terms = log_odds_dirichlet(tokens, df["label"].values)
    terms.to_csv(metrics_dir / "distinctive_terms.csv", index=False)
    tables["distinctive_terms"] = terms

    vocab = Counter(t for toks in tokens for t in toks)
    tables["preprocessing_effect"] = pd.DataFrame({
        "value": [
            df["text"].str.split().str.len().sum(),
            sum(len(t) for t in tokens),
            len(set(df["text"].str.lower().str.split().explode())),
            len(vocab),
            sum(len(t) == 0 for t in tokens),
        ]},
        index=["Raw whitespace tokens", "Tokens after preprocessing", "Raw vocabulary (lower-cased)",
               "Vocabulary after preprocessing", "Tweets empty after preprocessing"],
    )
    tables["preprocessing_effect"].to_csv(metrics_dir / "summary_preprocessing_effect.csv")

    fig_sentiment_distribution(df, figures_dir / "eda_sentiment_distribution.png")
    fig_sentiment_by_airline(df, figures_dir / "eda_sentiment_by_airline.png")
    fig_negative_reasons(df, figures_dir / "eda_negative_reasons.png")
    fig_reasons_by_airline(df, figures_dir / "eda_reasons_by_airline.png")
    fig_length_by_sentiment(df, figures_dir / "eda_length_by_sentiment.png")
    fig_distinctive_terms(terms, figures_dir / "eda_distinctive_terms.png")
    fig_daily_volume(df, figures_dir / "eda_daily_volume.png")
    proj = project_feature_space(pd.Series(TweetPreprocessor.join(tokens)), df["label"])
    fig_feature_space(proj, figures_dir / "feature_space.png")
    return tables
