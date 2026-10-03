"""Feature extraction: turning preprocessed text into numerical vectors.

Three representations are compared in this project:

1. **TF-IDF (word n-grams)** – sparse bag-of-n-grams weighted by term frequency ×
   inverse document frequency (Salton & Buckley, 1988). Bigrams capture short
   phrases such as "not happy" or "customer service".
2. **TF-IDF (word + character n-grams)** – character n-grams add robustness to the
   misspellings, elongations and creative spelling common in tweets.
3. **Word2Vec mean embeddings** – dense 100-d vectors learned on the training
   tweets (Mikolov et al., 2013) and averaged per tweet with TF-IDF weights.

Contextual sub-word embeddings (DistilRoBERTa) are produced inside the
Transformer model itself; see ``transformer_model.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import FeatureUnion

from . import config


def word_tfidf(ngram_range: tuple[int, int] = (1, 2), min_df: int = 2, **kwargs) -> TfidfVectorizer:
    """Word-level TF-IDF on already-preprocessed, space-joined tokens.

    ``sublinear_tf`` replaces raw counts with 1 + log(tf), which dampens the
    effect of a word repeated many times in one tweet ("delayed delayed delayed").
    """
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",  # tokens are already clean; keep all of them
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        **kwargs,
    )


def word_char_tfidf(char_ngram_range: tuple[int, int] = (2, 5)) -> FeatureUnion:
    """Concatenate word (1–2)-grams with character (2–5)-grams."""
    return FeatureUnion(
        [
            ("word", word_tfidf()),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngram_range,
                    min_df=3,
                    sublinear_tf=True,
                    max_features=60_000,
                ),
            ),
        ]
    )


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: space-joined tokens → TF-IDF-weighted mean Word2Vec.

    The embedding model is trained *only on the training fold* inside ``fit`` so no
    information leaks from validation/test tweets during cross-validation.

    Parameters
    ----------
    vector_size, window, min_count, epochs, sg:
        Standard gensim Word2Vec hyper-parameters (``sg=1`` → skip-gram, which
        works better than CBOW for small corpora and rare words).
    tfidf_weighting:
        Weight each word vector by its IDF so that frequent, uninformative words
        contribute less to the tweet vector.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 30,
        sg: int = 1,
        tfidf_weighting: bool = True,
        seed: int = config.SEED,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.tfidf_weighting = tfidf_weighting
        self.seed = seed

    def fit(self, X, y=None):
        sentences = [doc.split() for doc in X]
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            seed=self.seed,
            workers=1,  # single worker → deterministic vectors
        )
        if self.tfidf_weighting:
            tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(X)
            self.idf_ = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
        else:
            self.idf_ = {}
        return self

    def transform(self, X) -> np.ndarray:
        wv = self.model_.wv
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, doc in enumerate(X):
            words = [w for w in doc.split() if w in wv]
            if not words:
                continue  # empty tweet after preprocessing → zero vector
            weights = np.array([self.idf_.get(w, 1.0) for w in words])
            out[i] = np.average(wv[words], axis=0, weights=weights)
        return out


def project_feature_space(
    texts: pd.Series,
    labels: pd.Series,
    vectorizer: TfidfVectorizer | None = None,
    sample_size: int = 3000,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """2-D projections of the TF-IDF space for visualisation.

    Returns a DataFrame with linear (Truncated SVD / LSA) and non-linear (t-SNE on
    the 50 leading SVD components) coordinates for a class-balanced sample.
    """
    frame = pd.DataFrame({"text": texts.values, "label": labels.values})
    sample = pd.concat(
        g.sample(min(len(g), sample_size // 3), random_state=seed) for _, g in frame.groupby("label")
    )
    vectorizer = vectorizer or word_tfidf()
    X = vectorizer.fit_transform(sample["text"])
    svd50 = TruncatedSVD(n_components=50, random_state=seed).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(svd50)
    sample["svd_1"], sample["svd_2"] = svd50[:, 0], svd50[:, 1]
    sample["tsne_1"], sample["tsne_2"] = tsne[:, 0], tsne[:, 1]
    return sample.reset_index(drop=True)
