"""Tests for data cleaning, splitting, feature extraction and evaluation helpers."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loading import clean_dataset, load_raw, split_data, summary_statistics
from src.eda import log_odds_dirichlet
from src.evaluation import bootstrap_ci, compute_metrics, mcnemar_test
from src.features import Word2VecVectorizer, word_tfidf


@pytest.fixture(scope="module")
def cleaned():
    return clean_dataset(load_raw())


def test_cleaning_removes_duplicate_ids(cleaned):
    df, report = cleaned
    assert df["tweet_id"].is_unique
    assert report["raw_rows"] - report["duplicate_rows_removed"] == report["clean_rows"] == len(df)


def test_airline_label_corrected(cleaned):
    df, _ = cleaned
    assert "Delta" not in set(df["airline"])
    jetblue = df[df["airline"] == "JetBlue"]["text"].str.lower()
    assert jetblue.str.contains("@jetblue").mean() > 0.85


def test_labels_are_encoded_consistently(cleaned):
    df, _ = cleaned
    decoded = df["label"].map(config.ID2LABEL)
    assert (decoded == df["airline_sentiment"].astype(str)).all()


def test_split_is_disjoint_stratified_and_complete(cleaned):
    df, _ = cleaned
    s = split_data(df)
    ids = [set(p["tweet_id"]) for p in (s.train, s.val, s.test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(df)
    overall = df["label"].value_counts(normalize=True)
    for part in (s.train, s.val, s.test):
        assert np.allclose(part["label"].value_counts(normalize=True)[overall.index], overall, atol=0.01)


def test_split_is_reproducible(cleaned):
    df, _ = cleaned
    assert split_data(df).test["tweet_id"].tolist() == split_data(df).test["tweet_id"].tolist()


def test_summary_statistics_tables(cleaned):
    df, _ = cleaned
    tables = summary_statistics(df)
    assert tables["sentiment"]["count"].sum() == len(df)
    assert set(tables) >= {"overview", "sentiment", "by_airline", "negative_reasons"}


DOCS = ["great flight thank crew", "delay hour hold bad", "flight tomorrow dm please",
        "love great crew", "cancel delay bad bag", "flight dm follow"] * 5


def test_word_tfidf_bigrams():
    X = word_tfidf(ngram_range=(1, 2), min_df=1).fit(DOCS)
    assert "great flight" in X.vocabulary_


def test_word2vec_vectorizer_shape_and_empty_doc():
    vec = Word2VecVectorizer(vector_size=16, epochs=2, min_count=1).fit(DOCS)
    X = vec.transform(DOCS + ["", "unseenword"])
    assert X.shape == (len(DOCS) + 2, 16)
    assert np.allclose(X[-2:], 0)


def test_compute_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 0])
    scores = np.eye(3)[y]
    m = compute_metrics(y, y, scores)
    assert m["accuracy"] == m["f1_macro"] == m["roc_auc_macro_ovr"] == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    pred = np.where(rng.random(500) < 0.8, y, (y + 1) % 3)
    ci = bootstrap_ci(y, pred, n_resamples=200)
    acc = (y == pred).mean()
    assert ci["accuracy_ci_low"] <= acc <= ci["accuracy_ci_high"]


def test_mcnemar_identical_models_not_significant():
    y = np.array([0, 1, 2] * 20)
    res = mcnemar_test(y, y, y)
    assert res["a_only_correct"] == res["b_only_correct"] == 0
    assert res["p_value"] == pytest.approx(1.0)


def test_log_odds_ranks_class_specific_words():
    tokens = [["thank", "great"]] * 20 + [["delay", "bad"]] * 20 + [["dm", "flight"]] * 20
    labels = np.array([2] * 20 + [0] * 20 + [1] * 20)
    terms = log_odds_dirichlet(tokens, labels, top_n=2, min_count=1)
    assert set(terms[terms["sentiment"] == "positive"]["term"]) == {"thank", "great"}
    assert set(terms[terms["sentiment"] == "negative"]["term"]) == {"delay", "bad"}
