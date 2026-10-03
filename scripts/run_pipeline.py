"""End-to-end, reproducible experiment runner.

Usage (from the project root)::

    python scripts/run_pipeline.py                       # every stage
    python scripts/run_pipeline.py --stages classical evaluate
    python scripts/run_pipeline.py --skip-transformer    # ≈5 min instead of ≈1 h on CPU

Stages (run in this order; each writes its outputs to ``results/``):

    eda          dataset summary tables + exploratory figures
    classical    grid-searched NB / LR / SVM / Word2Vec models → test predictions
    ablation     5-fold CV study of each preprocessing step (uses the tuned SVM)
    rnn          BiLSTM training with early stopping → test predictions
    transformer  DistilRoBERTa fine-tuning → test predictions
    evaluate     metrics, bootstrap CIs, McNemar tests, confusion matrices, ROC curves
    insights     business-facing analysis (key terms, airline KPIs, error analysis)

Stages communicate only through files in ``results/`` and ``models/``, so a slow
stage (the Transformer) can be run on its own and the rest re-run cheaply.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data_loading import DataSplits, clean_dataset, load_raw, split_data  # noqa: E402
from src.preprocessing import TweetPreprocessor, light_clean  # noqa: E402

STAGES = ["eda", "classical", "ablation", "rnn", "transformer", "evaluate", "insights"]
MODEL_INFO_DIR = config.METRICS_DIR / "models"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_model_infos() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(MODEL_INFO_DIR.glob("*.json"))}


class Context:
    """Lazily computed dataset, splits and preprocessed text shared by stages."""

    def __init__(self) -> None:
        raw = load_raw()
        self.df, self.cleaning_report = clean_dataset(raw)
        self.splits: DataSplits = split_data(self.df)
        self._cache: dict[str, dict[str, list]] = {}
        sizes = {k: len(getattr(self.splits, k)) for k in ("train", "val", "test")}
        log(f"Loaded {len(self.df):,} tweets after cleaning; split sizes {sizes}")
        pd.DataFrame(
            [(tid, part) for part in ("train", "val", "test") for tid in getattr(self.splits, part)["tweet_id"]],
            columns=["tweet_id", "split"],
        ).to_csv(config.METRICS_DIR / "split_assignment.csv", index=False)

    def tokens(self, variant: str = "full") -> dict[str, list]:
        """Preprocessed tokens per split. ``full`` = assignment pipeline; ``sequence`` keeps stop words."""
        if variant not in self._cache:
            pp = {
                "full": TweetPreprocessor(),
                "sequence": TweetPreprocessor(remove_stopwords=False),
            }[variant]
            self._cache[variant] = {
                part: pp.process_corpus(getattr(self.splits, part)["text"]) for part in ("train", "val", "test")
            }
        return self._cache[variant]

    def joined(self, part: str, variant: str = "full") -> list[str]:
        return TweetPreprocessor.join(self.tokens(variant)[part])


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_eda(ctx: Context) -> None:
    from src import eda

    all_tokens = TweetPreprocessor().process_corpus(ctx.df["text"])
    eda.run_eda(ctx.df, ctx.cleaning_report, all_tokens, config.FIGURES_DIR, config.METRICS_DIR)
    write_json(ctx.cleaning_report, config.METRICS_DIR / "cleaning_report.json")


N_JOBS = 2  # overridden by --n-jobs; parallel CV workers each hold a copy of the data
MODEL_FILTER: set[str] = set()  # overridden by --models; empty = every classical model


def stage_classical(ctx: Context) -> None:
    from src.classical_models import decision_scores, get_model_specs, tune_and_fit
    from src.evaluation import save_predictions

    # Classical models are tuned with 5-fold CV on train+val (no early stopping needed).
    X_fit = ctx.joined("train") + ctx.joined("val")
    y_fit = np.concatenate([ctx.splits.train["label"].values, ctx.splits.val["label"].values])
    X_test, test = ctx.joined("test"), ctx.splits.test

    for spec in get_model_specs():
        if MODEL_FILTER and spec.slug not in MODEL_FILTER:
            continue
        log(f"Tuning {spec.name} ({spec.features}) …")
        model, info = tune_and_fit(spec, X_fit, y_fit, n_jobs=N_JOBS)
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        info["inference_ms_per_1k"] = round((time.perf_counter() - start) * 1000 / len(X_test) * 1000, 1)
        info["family"] = "Classical ML"
        save_predictions(spec.slug, test["tweet_id"], test["label"], y_pred, decision_scores(model, X_test))
        joblib.dump(model, config.MODELS_DIR / f"{spec.slug}.joblib")
        write_json(info, MODEL_INFO_DIR / f"{spec.slug}.json")
        log(f"  best CV macro-F1 {info['cv_macro_f1_mean']:.4f} ± {info['cv_macro_f1_std']:.4f} "
            f"params={info['best_params']} ({info['tuning_seconds']}s)")


def stage_ablation(ctx: Context) -> None:
    """Contribution of each preprocessing step, measured by 5-fold CV on train+val."""
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    from src.classical_models import cross_val_macro_f1
    from src.features import word_tfidf

    info_path = MODEL_INFO_DIR / "svm_tfidf.json"
    params = json.loads(info_path.read_text())["best_params"] if info_path.exists() else {}
    C = params.get("clf__C", 0.25)
    cw = params.get("clf__class_weight")
    ngram = tuple(params.get("tfidf__ngram_range", (1, 2)))

    fit_df = ctx.splits.train_full
    texts, y = fit_df["text"].tolist(), fit_df["label"].values
    variants = {
        "Raw text (lower-cased only)": (lambda: [t.lower() for t in texts], True),
        "Normalised + tokenised (keep stop words, no lemmatisation)":
            (lambda: TweetPreprocessor.join(TweetPreprocessor(remove_stopwords=False, lemmatize=False).process_corpus(texts)), False),
        "+ stop-word removal (negations kept)":
            (lambda: TweetPreprocessor.join(TweetPreprocessor(lemmatize=False).process_corpus(texts)), False),
        "+ lemmatisation (full pipeline – used)":
            (lambda: TweetPreprocessor.join(TweetPreprocessor().process_corpus(texts)), False),
        "Full pipeline but negations removed as stop words":
            (lambda: TweetPreprocessor.join(TweetPreprocessor(keep_negations=False).process_corpus(texts)), False),
        "Lemmatisation without stop-word removal":
            (lambda: TweetPreprocessor.join(TweetPreprocessor(remove_stopwords=False).process_corpus(texts)), False),
    }
    rows = []
    for name, (make_docs, default_tokenizer) in variants.items():
        docs = make_docs()

        def build(default_tokenizer=default_tokenizer):
            vec = word_tfidf(ngram_range=ngram)
            if default_tokenizer:
                vec.set_params(token_pattern=r"(?u)\b\w\w+\b")
            return Pipeline([("tfidf", vec), ("clf", LinearSVC(C=C, class_weight=cw, random_state=config.SEED, max_iter=10_000))])

        mean, std = cross_val_macro_f1(build, docs, y, n_jobs=N_JOBS)
        vocab = len(build().named_steps["tfidf"].fit(docs).vocabulary_)
        rows.append({"variant": name, "cv_macro_f1_mean": mean, "cv_macro_f1_std": std, "vocabulary_size": vocab})
        log(f"  {name:<60s} macro-F1 {mean:.4f} ± {std:.4f}  vocab {vocab:,}")
    pd.DataFrame(rows).to_csv(config.METRICS_DIR / "preprocessing_ablation.csv", index=False)


def stage_rnn(ctx: Context) -> None:
    import torch

    from src.evaluation import plot_training_history, save_predictions
    from src.rnn_model import RNNConfig, predict_proba, train_bilstm

    torch.set_num_threads(8)
    tok = ctx.tokens("sequence")
    cfg = RNNConfig()
    log("Training BiLSTM …")
    model, vocab, info = train_bilstm(tok["train"], ctx.splits.train["label"].values,
                                      tok["val"], ctx.splits.val["label"].values, cfg)
    start = time.perf_counter()
    probs = predict_proba(model, vocab, tok["test"], cfg.max_len)
    info["inference_ms_per_1k"] = round((time.perf_counter() - start) * 1000 / len(probs) * 1000, 1)
    test = ctx.splits.test
    save_predictions("bilstm", test["tweet_id"], test["label"], probs.argmax(1), probs)
    torch.save({"state_dict": model.state_dict(), "itos": vocab.itos, "config": info["config"]},
               config.MODELS_DIR / "bilstm.pt")
    info.update(slug="bilstm", name="BiLSTM (RNN)", algorithm="Bidirectional LSTM",
                features="Word2Vec-initialised embeddings (fine-tuned)", family="Deep learning (RNN)")
    write_json(info, MODEL_INFO_DIR / "bilstm.json")
    plot_training_history(info["history"], "BiLSTM training curves", config.FIGURES_DIR / "training_bilstm.png")


def stage_transformer(ctx: Context, epochs: int) -> None:
    from src.evaluation import plot_training_history, save_predictions
    from src.transformer_model import TransformerConfig, predict_proba, train_transformer

    s = ctx.splits
    cfg = TransformerConfig(epochs=epochs)
    log(f"Fine-tuning {cfg.model_name} for {epochs} epochs on CPU (this is the slow stage) …")
    model, tokenizer, info = train_transformer(
        [light_clean(t) for t in s.train["text"]], s.train["label"].values,
        [light_clean(t) for t in s.val["text"]], s.val["label"].values,
        cfg, save_dir=config.MODELS_DIR / "distilroberta",
    )
    start = time.perf_counter()
    probs = predict_proba(model, tokenizer, [light_clean(t) for t in s.test["text"]], cfg.max_length)
    info["inference_ms_per_1k"] = round((time.perf_counter() - start) * 1000 / len(probs) * 1000, 1)
    save_predictions("distilroberta", s.test["tweet_id"], s.test["label"], probs.argmax(1), probs)
    info.update(slug="distilroberta", name="DistilRoBERTa (Transformer)", algorithm="Transformer (fine-tuned)",
                features="Contextual sub-word embeddings (BPE)", family="Deep learning (Transformer)")
    write_json(info, MODEL_INFO_DIR / "distilroberta.json")
    plot_training_history(info["history"], "DistilRoBERTa fine-tuning curves", config.FIGURES_DIR / "training_distilroberta.png")


def stage_evaluate(ctx: Context) -> None:
    from src import evaluation as ev
    from src.plotting import MODEL_ORDER

    preds = ev.load_predictions()
    ev.validate_predictions(preds, ctx.splits.test)
    infos = read_model_infos()
    names = {slug: info["name"] for slug, info in infos.items()}
    order = [m for m in MODEL_ORDER if m in preds]

    table = ev.comparison_table(preds, names)
    for col in ("family", "algorithm", "features", "cv_macro_f1_mean", "best_val_macro_f1", "inference_ms_per_1k"):
        table[col] = table["slug"].map(lambda s, c=col: infos.get(s, {}).get(c))
    table.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    log("\n" + table[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro_ovr"]]
        .round(4).to_string(index=False))

    best = table.iloc[0]["slug"]
    ev.significance_table(preds, best, names).to_csv(config.METRICS_DIR / "mcnemar_tests.csv", index=False)
    per_class = pd.concat(
        {names.get(s, s): ev.per_class_report(preds[s]["y_true"], preds[s]["y_pred"]) for s in order},
        names=["model", "class"],
    )
    per_class.to_csv(config.METRICS_DIR / "per_class_metrics.csv")

    fig_dir = config.FIGURES_DIR
    ev.plot_model_comparison(table, fig_dir / "model_comparison.png")
    ev.plot_confusion_grid(preds, names, fig_dir / "confusion_matrices.png", order)
    ev.plot_confusion_matrix(preds[best]["y_true"], preds[best]["y_pred"], f"{names[best]} – test set")
    import matplotlib.pyplot as plt

    from src.plotting import save
    save(plt.gcf(), fig_dir / "confusion_matrix_best.png")
    ev.plot_roc_per_class(preds[best], f"ROC curves – {names[best]}", fig_dir / "roc_best_model.png")
    ev.plot_roc_by_model(preds, names, fig_dir / "roc_all_models.png", order)
    ev.plot_per_class_f1(preds, names, fig_dir / "per_class_f1.png", order)
    write_json({"best_model": best, "best_model_name": names[best]}, config.METRICS_DIR / "best_model.json")


def stage_insights(ctx: Context) -> None:
    from src import insights
    from src.evaluation import load_predictions

    preds = load_predictions()
    best = json.loads((config.METRICS_DIR / "best_model.json").read_text())["best_model"]
    linear = {slug: joblib.load(config.MODELS_DIR / f"{slug}.joblib") for slug in ("lr_tfidf", "svm_tfidf")
              if (config.MODELS_DIR / f"{slug}.joblib").exists()}
    from src.evaluation import validate_predictions
    validate_predictions(preds, ctx.splits.test)
    insights.run_insights(ctx.splits.test, preds, best, linear, config.FIGURES_DIR, config.METRICS_DIR)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--skip-transformer", action="store_true", help="skip DistilRoBERTa fine-tuning")
    parser.add_argument("--transformer-epochs", type=int, default=3)
    parser.add_argument("--models", nargs="+", metavar="SLUG",
                        help="classical stage only: re-tune just these models (e.g. lr_tfidf svm_word_char)")
    parser.add_argument("--n-jobs", type=int, default=2,
                        help="parallel CV workers for classical models (each needs ~1 GB RAM; -1 = all cores)")
    args = parser.parse_args()
    global N_JOBS, MODEL_FILTER
    N_JOBS = args.n_jobs
    MODEL_FILTER = set(args.models or [])

    config.ensure_dirs()
    config.set_seed()
    stages = [s for s in STAGES if s in args.stages and not (s == "transformer" and args.skip_transformer)]
    ctx = Context()
    for stage in stages:
        log(f"=== Stage: {stage} ===")
        t0 = time.perf_counter()
        if stage == "transformer":
            stage_transformer(ctx, args.transformer_epochs)
        else:
            globals()[f"stage_{stage}"](ctx)
        log(f"=== {stage} finished in {time.perf_counter() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
