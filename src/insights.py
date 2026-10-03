"""Business-facing interpretation of the trained models.

Questions answered (all on the held-out test set)
-------------------------------------------------
1. *Which words drive each prediction?* – coefficients of the linear TF-IDF model.
2. *Can the model replace manual tagging for airline-level KPIs?* – actual vs.
   predicted share of negative tweets and Net Sentiment Score per airline.
3. *Which complaint types does the model catch?* – recall on negative tweets by
   annotated complaint reason.
4. *Where does the model fail, and is it the model or the labels?* – accuracy by
   annotator-confidence band, plus the most confident errors.
5. *How should a triage queue be thresholded?* – precision/recall trade-off for
   detecting negative tweets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import config
from .evaluation import SCORE_COLUMNS
from .plotting import MODEL_COLORS, TEXT_PRIMARY, TEXT_SECONDARY, save

CONFIDENCE_BINS = [0.0, 0.6, 0.8, 0.999, 1.0]
CONFIDENCE_LABELS = ["< 0.60", "0.60–0.80", "0.80–0.99", "1.00 (maximum score)"]


def _merge(test: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    return test.merge(pred, on="tweet_id", validate="one_to_one")


# --------------------------------------------------------------------------- #
# 1. Model explanation
# --------------------------------------------------------------------------- #
def top_coefficients(pipeline, n: int = 15) -> pd.DataFrame:
    """Highest-weighted n-grams per class from a fitted TF-IDF → linear-model pipeline."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    names = vec.get_feature_names_out()
    rows = []
    for k, label in enumerate(config.LABELS):
        for idx in np.argsort(-clf.coef_[k])[:n]:
            rows.append({"sentiment": label, "feature": names[idx], "coefficient": float(clf.coef_[k, idx])})
    return pd.DataFrame(rows)


def fig_top_coefficients(coefs: pd.DataFrame, title: str, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, label in zip(axes, config.LABELS):
        t = coefs[coefs["sentiment"] == label].sort_values("coefficient")
        ax.barh(t["feature"], t["coefficient"], color=config.LABEL_COLORS[label], height=0.65)
        ax.set_title(label.capitalize())
        ax.set_xlabel("Model coefficient")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return save(fig, path)


# --------------------------------------------------------------------------- #
# 2. Airline KPIs
# --------------------------------------------------------------------------- #
def airline_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """% negative and Net Sentiment Score (NSS = % positive − % negative), actual vs. predicted."""
    def shares(col):
        s = pd.crosstab(df["airline"], df[col], normalize="index").reindex(columns=range(3), fill_value=0) * 100
        return s[0], s[2]

    neg_a, pos_a = shares("y_true")
    neg_p, pos_p = shares("y_pred")
    out = pd.DataFrame({
        "n_test_tweets": df["airline"].value_counts(),
        "actual_pct_negative": neg_a, "predicted_pct_negative": neg_p,
        "actual_nss": pos_a - neg_a, "predicted_nss": pos_p - neg_p,
    })
    out["abs_error_nss"] = (out["actual_nss"] - out["predicted_nss"]).abs()
    out["actual_rank"] = out["actual_nss"].rank(ascending=False).astype(int)
    out["predicted_rank"] = out["predicted_nss"].rank(ascending=False).astype(int)
    return out.sort_values("actual_nss").round(2)


def fig_airline_kpis(kpis: pd.DataFrame, model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(kpis))
    for yi, (a, p) in enumerate(zip(kpis["actual_nss"], kpis["predicted_nss"])):
        ax.plot([a, p], [yi, yi], color="#c9c8c2", lw=2, zorder=1)
    ax.scatter(kpis["actual_nss"], y, s=70, color=TEXT_PRIMARY, zorder=2, label="Actual (human labels)")
    ax.scatter(kpis["predicted_nss"], y, s=70, color="#2a78d6", edgecolor="white", linewidth=1.5, zorder=3,
               label=f"Predicted ({model_name})")
    ax.axvline(0, color="#b9b8b2", lw=1)
    ax.set_yticks(y, kpis.index)
    ax.set_xlabel("Net Sentiment Score  (% positive − % negative)")
    ax.set_title("Airline Net Sentiment Score: model vs. human labels (test set)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return save(fig, path)


# --------------------------------------------------------------------------- #
# 3–4. Error analysis
# --------------------------------------------------------------------------- #
def recall_by_reason(df: pd.DataFrame) -> pd.DataFrame:
    neg = df[(df["y_true"] == 0) & df["negativereason"].notna()]
    return (neg.assign(correct=neg["y_pred"] == 0)
            .groupby("negativereason")["correct"].agg(n="size", recall="mean")
            .sort_values("recall").round(3))


def accuracy_by_confidence(merged_by_model: dict[str, pd.DataFrame], names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for slug, df in merged_by_model.items():
        band = pd.cut(df["airline_sentiment_confidence"], CONFIDENCE_BINS, labels=CONFIDENCE_LABELS, include_lowest=True)
        g = df.assign(band=band, correct=df["y_true"] == df["y_pred"]).groupby("band", observed=False)["correct"]
        for b, (n, acc) in g.agg(["size", "mean"]).iterrows():
            rows.append({"model": names.get(slug, slug), "slug": slug, "confidence_band": b, "n": int(n), "accuracy": acc})
    return pd.DataFrame(rows)


def fig_accuracy_by_confidence(table: pd.DataFrame, path: Path) -> Path:
    slugs = list(dict.fromkeys(table["slug"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / len(slugs)
    x = np.arange(len(CONFIDENCE_LABELS))
    for i, slug in enumerate(slugs):
        t = table[table["slug"] == slug].set_index("confidence_band").reindex(CONFIDENCE_LABELS)
        bars = ax.bar(x + i * width - 0.4 + width / 2, t["accuracy"], width * 0.92, color=MODEL_COLORS.get(slug),
                      label=t["model"].iloc[0])
        for b, v in zip(bars, t["accuracy"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=TEXT_PRIMARY)
    n = table[table["slug"] == slugs[0]].set_index("confidence_band").reindex(CONFIDENCE_LABELS)["n"]
    ax.set_xticks(x, [f"{c}\n(n={k:,})" for c, k in zip(CONFIDENCE_LABELS, n)])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Annotator confidence in the gold label")
    ax.set_title("Model accuracy by annotation confidence")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    return save(fig, path)


def confident_errors(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Misclassified tweets on which the model was most confident (candidate label noise)."""
    wrong = df[df["y_true"] != df["y_pred"]].copy()
    wrong["model_confidence"] = wrong[SCORE_COLUMNS].max(axis=1)
    wrong["true"] = wrong["y_true"].map(config.ID2LABEL)
    wrong["predicted"] = wrong["y_pred"].map(config.ID2LABEL)
    cols = ["tweet_id", "airline", "text", "true", "predicted", "model_confidence", "airline_sentiment_confidence"]
    return wrong.sort_values("model_confidence", ascending=False)[cols].head(n)


# --------------------------------------------------------------------------- #
# 5. Triage threshold
# --------------------------------------------------------------------------- #
def negative_triage(df: pd.DataFrame, targets=(0.90, 0.95, 0.98)) -> tuple[pd.DataFrame, dict]:
    """Exploratory PR operating points selected on the supplied labelled data.

    When supplied test predictions, these are descriptive test-set results, not
    independently evaluated deployment thresholds. Select deployment thresholds
    on validation/out-of-fold scores, then assess them on untouched data.
    SVM scores are uncalibrated margins, not confidence probabilities.
    """
    y = (df["y_true"] == 0).astype(int)
    s = df["score_negative"]
    precision, recall, thresholds = precision_recall_curve(y, s)
    rows = []
    for target in targets:
        ok = np.where(recall[:-1] >= target)[0]
        i = ok[-1]  # highest threshold that still reaches the recall target
        flagged = int((s >= thresholds[i]).sum())
        rows.append({"target_recall": target, "threshold": float(thresholds[i]), "precision": float(precision[i]),
                     "recall": float(recall[i]), "share_of_tweets_flagged": flagged / len(df)})
    curve = {"precision": precision, "recall": recall, "ap": average_precision_score(y, s), "base_rate": y.mean()}
    return pd.DataFrame(rows), curve


def fig_negative_pr(curve: dict, points: pd.DataFrame, model_name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(curve["recall"], curve["precision"], color=config.LABEL_COLORS["negative"],
            label=f"{model_name} (AP = {curve['ap']:.3f})")
    ax.axhline(curve["base_rate"], ls="--", lw=1, color="#b9b8b2", label=f"Base rate ({curve['base_rate']:.0%})")
    for _, r in points.iterrows():
        ax.scatter(r["recall"], r["precision"], s=50, color=TEXT_PRIMARY, zorder=3)
        ax.annotate(f"recall {r['recall']:.0%} → precision {r['precision']:.0%}", (r["recall"], r["precision"]),
                    textcoords="offset points", xytext=(-8, -14), ha="right", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.5, 1.01)
    ax.set_xlabel("Recall (share of complaints caught)")
    ax.set_ylabel("Precision (share of flagged tweets that are complaints)")
    ax.set_title("Complaint triage: precision–recall for 'negative'")
    ax.legend(loc="lower left")
    return save(fig, path)


# --------------------------------------------------------------------------- #
def run_insights(test: pd.DataFrame, predictions: dict[str, pd.DataFrame], best_slug: str,
                 linear_models: dict, figures_dir: Path, metrics_dir: Path) -> dict[str, pd.DataFrame]:
    import json

    infos = {p.stem: json.loads(p.read_text()) for p in (metrics_dir / "models").glob("*.json")}
    names = {k: v["name"] for k, v in infos.items()}
    best = _merge(test, predictions[best_slug])
    out = {}

    if "lr_tfidf" in linear_models:
        out["top_coefficients"] = top_coefficients(linear_models["lr_tfidf"])
        out["top_coefficients"].to_csv(metrics_dir / "top_coefficients_lr.csv", index=False)
        fig_top_coefficients(out["top_coefficients"], "What drives the Logistic Regression model (top TF-IDF coefficients)",
                             figures_dir / "insight_top_coefficients.png")

    out["airline_kpis"] = airline_kpis(best)
    out["airline_kpis"].to_csv(metrics_dir / "airline_kpis.csv")
    fig_airline_kpis(out["airline_kpis"], names.get(best_slug, best_slug), figures_dir / "insight_airline_nss.png")

    out["recall_by_reason"] = recall_by_reason(best)
    out["recall_by_reason"].to_csv(metrics_dir / "recall_by_negative_reason.csv")

    compare = [s for s in (best_slug, "svm_tfidf", "nb_tfidf") if s in predictions]
    compare = list(dict.fromkeys(compare))
    out["accuracy_by_confidence"] = accuracy_by_confidence({s: _merge(test, predictions[s]) for s in compare}, names)
    out["accuracy_by_confidence"].to_csv(metrics_dir / "accuracy_by_confidence.csv", index=False)
    fig_accuracy_by_confidence(out["accuracy_by_confidence"], figures_dir / "insight_accuracy_by_confidence.png")

    out["confident_errors"] = confident_errors(best)
    out["confident_errors"].to_csv(metrics_dir / "confident_errors.csv", index=False)

    out["triage"], curve = negative_triage(best)
    out["triage"].to_csv(metrics_dir / "negative_triage_operating_points.csv", index=False)
    fig_negative_pr(curve, out["triage"], names.get(best_slug, best_slug), figures_dir / "insight_negative_pr.png")
    return out
