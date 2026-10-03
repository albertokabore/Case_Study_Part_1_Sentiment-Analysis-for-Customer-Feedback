"""Classical machine-learning sentiment models with cross-validated tuning.

Each model is a scikit-learn ``Pipeline`` (vectoriser → classifier), so feature
extraction is re-fitted inside every cross-validation fold and no vocabulary or
IDF statistics leak from the held-out fold.

Models
------
* Multinomial Naive Bayes + word TF-IDF – fast probabilistic baseline
  (McCallum & Nigam, 1998).
* Logistic Regression + word TF-IDF – calibrated linear model with interpretable
  coefficients.
* Linear SVM + word TF-IDF – max-margin linear classifier, a strong text baseline
  (Joachims, 1998).
* Linear SVM + word & character TF-IDF – adds sub-word robustness.
* Logistic Regression + Word2Vec – tests dense embeddings vs. sparse TF-IDF.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config
from .features import Word2VecVectorizer, word_char_tfidf, word_tfidf


@dataclass
class ModelSpec:
    """Declarative description of a model and its hyper-parameter search space."""

    slug: str
    name: str
    algorithm: str
    features: str
    build: Callable[[], Pipeline]
    param_grid: dict = field(default_factory=dict)


def _lr() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, random_state=config.SEED)


def _svm() -> LinearSVC:
    return LinearSVC(random_state=config.SEED, max_iter=10_000)


def get_model_specs() -> list[ModelSpec]:
    """All classical models compared in the study, with their search grids."""
    class_weight = [None, "balanced"]
    return [
        ModelSpec(
            slug="nb_tfidf",
            name="Naive Bayes",
            algorithm="Multinomial Naive Bayes",
            features="TF-IDF (word n-grams)",
            build=lambda: Pipeline([("tfidf", word_tfidf()), ("clf", MultinomialNB())]),
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__alpha": [0.05, 0.1, 0.3, 0.5, 1.0],
                "clf__fit_prior": [True, False],
            },
        ),
        ModelSpec(
            slug="lr_tfidf",
            name="Logistic Regression",
            algorithm="Logistic Regression (multinomial)",
            features="TF-IDF (word n-grams)",
            build=lambda: Pipeline([("tfidf", word_tfidf()), ("clf", _lr())]),
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.1, 0.25, 0.5, 1, 2, 5],
                "clf__class_weight": class_weight,
            },
        ),
        ModelSpec(
            slug="svm_tfidf",
            name="Linear SVM",
            algorithm="Linear Support Vector Machine",
            features="TF-IDF (word n-grams)",
            build=lambda: Pipeline([("tfidf", word_tfidf()), ("clf", _svm())]),
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.05, 0.1, 0.25, 0.5, 1.0],
                "clf__class_weight": class_weight,
            },
        ),
        ModelSpec(
            slug="svm_word_char",
            name="Linear SVM (word+char)",
            algorithm="Linear Support Vector Machine",
            features="TF-IDF (word 1-2 + char 2-5 n-grams)",
            build=lambda: Pipeline([("tfidf", word_char_tfidf()), ("clf", _svm())]),
            param_grid={
                "clf__C": [0.01, 0.025, 0.05, 0.1, 0.25],
                "clf__class_weight": class_weight,
            },
        ),
        ModelSpec(
            slug="lr_word2vec",
            name="Logistic Regression (Word2Vec)",
            algorithm="Logistic Regression (multinomial)",
            features="Word2Vec skip-gram, 100-d, IDF-weighted mean",
            build=lambda: Pipeline(
                [
                    ("w2v", Word2VecVectorizer()),
                    ("scale", StandardScaler()),
                    ("clf", _lr()),
                ]
            ),
            param_grid={
                "clf__C": [0.01, 0.1, 1.0],
                "clf__class_weight": class_weight,
            },
        ),
    ]


def tune_and_fit(
    spec: ModelSpec,
    X: pd.Series | list[str],
    y: np.ndarray,
    cv_folds: int = config.CV_FOLDS,
    n_jobs: int = 2,
) -> tuple[Pipeline, dict]:
    """Grid-search ``spec`` with stratified k-fold CV (macro-F1) and refit on all of X.

    Macro-F1 is the selection criterion because the classes are imbalanced
    (≈63 % negative); accuracy alone would reward ignoring the minority classes.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(
        spec.build(),
        spec.param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    start = time.perf_counter()
    search.fit(list(X), y)
    elapsed = time.perf_counter() - start

    best_idx = search.best_index_
    info = {
        "slug": spec.slug,
        "name": spec.name,
        "algorithm": spec.algorithm,
        "features": spec.features,
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "cv_macro_f1_mean": float(search.best_score_),
        "cv_macro_f1_std": float(search.cv_results_["std_test_score"][best_idx]),
        "n_candidates": len(search.cv_results_["params"]),
        "param_grid": {k: [_jsonable(v) for v in values] for k, values in spec.param_grid.items()},
        "tuning_seconds": round(elapsed, 1),
    }
    return search.best_estimator_, info


def decision_scores(model: Pipeline, X) -> np.ndarray:
    """Per-class scores for ROC analysis (probabilities if available, else margins)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(list(X))
    return model.decision_function(list(X))


def cross_val_macro_f1(build: Callable[[], Pipeline], X, y, cv_folds: int = config.CV_FOLDS,
                       n_jobs: int = 2) -> tuple[float, float]:
    """Mean ± std macro-F1 of a fixed pipeline (used for the preprocessing ablation)."""
    from sklearn.model_selection import cross_val_score

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.SEED)
    scores = cross_val_score(build(), list(X), y, scoring="f1_macro", cv=cv, n_jobs=n_jobs)
    return float(scores.mean()), float(scores.std())


def _jsonable(value):
    """Make grid values (tuples, numpy scalars) JSON serialisable."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
