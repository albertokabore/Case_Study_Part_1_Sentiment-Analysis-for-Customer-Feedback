"""Data loading, cleaning and train/validation/test splitting.

The raw file is the Kaggle "Twitter US Airline Sentiment" CSV (CrowdFlower, 2015).
Two data-quality issues are handled here, both documented in the research paper:

1. **Duplicate tweets** – 155 rows re-use an existing ``tweet_id`` (the tweet was
   collected twice); 18 of those pairs carry *conflicting* sentiment labels. We keep
   one row per ``tweet_id``, preferring the annotation with the highest
   ``airline_sentiment_confidence``.
2. **Mislabelled airline** – rows tagged ``Delta`` are, in ≈90 % of cases, addressed
   to ``@JetBlue``. The ``airline`` column is corrected (``Delta`` → ``JetBlue``) so
   airline-level insights are attributed to the right carrier. The text and the
   sentiment labels are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

# Only the columns needed for modelling and business analysis are kept.
USE_COLUMNS = [
    "tweet_id",
    "airline_sentiment",
    "airline_sentiment_confidence",
    "negativereason",
    "negativereason_confidence",
    "airline",
    "retweet_count",
    "text",
    "tweet_created",
]


def load_raw(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as distributed (no cleaning)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Tweets.csv in the project root."
        )
    return pd.read_csv(path)


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """De-duplicate, correct the airline label and type-cast columns.

    Returns
    -------
    df : DataFrame
        Clean dataset, one row per tweet.
    report : dict
        Counts describing each cleaning step (reported in the paper).
    """
    missing = set(USE_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = raw[USE_COLUMNS].copy()
    report = {"raw_rows": len(df)}

    # 1. Duplicates: keep the most confident annotation for each tweet_id.
    dup_mask = df["tweet_id"].duplicated(keep=False)
    conflicting = (
        df[dup_mask].groupby("tweet_id")["airline_sentiment"].nunique().gt(1).sum()
    )
    df = df.sort_values("airline_sentiment_confidence", ascending=False)
    df = df.drop_duplicates("tweet_id", keep="first")
    report["duplicate_rows_removed"] = report["raw_rows"] - len(df)
    report["duplicate_ids_with_conflicting_labels"] = int(conflicting)

    # 2. Airline label correction (Delta -> JetBlue), recorded for transparency.
    report["airline_rows_relabelled"] = int(
        df["airline"].isin(config.AIRLINE_LABEL_CORRECTIONS).sum()
    )
    df["airline"] = df["airline"].replace(config.AIRLINE_LABEL_CORRECTIONS)

    # 3. Types and ordering.
    df = df[df["text"].notna() & df["airline_sentiment"].isin(config.LABELS)]
    df["tweet_created"] = pd.to_datetime(df["tweet_created"], utc=True)
    df["airline_sentiment"] = pd.Categorical(
        df["airline_sentiment"], categories=config.LABELS
    )
    df["label"] = df["airline_sentiment"].cat.codes.astype(int)
    df = df.sort_values("tweet_id").reset_index(drop=True)

    report["clean_rows"] = len(df)
    return df, report


def load_dataset(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load and clean in one call."""
    df, _ = clean_dataset(load_raw(path))
    return df


@dataclass(frozen=True)
class DataSplits:
    """Stratified train / validation / test partitions."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_full(self) -> pd.DataFrame:
        """Train + validation, used to fit classical models with cross-validation."""
        return pd.concat([self.train, self.val]).sort_values("tweet_id")


def split_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.SEED,
) -> DataSplits:
    """Stratified 70 / 15 / 15 split (by default) on the sentiment label.

    The test set is held out for the final comparison of *all* models. Classical
    models are tuned by cross-validation on train+val; neural models train on
    ``train`` and use ``val`` for early stopping. All models therefore see the
    same test tweets, which makes the comparison fair.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val["label"],
        random_state=seed,
    )
    return DataSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def summary_statistics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Descriptive tables used in the Data Collection section of the paper."""
    word_counts = df["text"].str.split().str.len()
    overview = pd.DataFrame(
        {
            "value": [
                len(df),
                df["airline"].nunique(),
                df["tweet_created"].min().date().isoformat(),
                df["tweet_created"].max().date().isoformat(),
                round(word_counts.mean(), 1),
                int(word_counts.median()),
                round(df["airline_sentiment_confidence"].mean(), 3),
                round((df["airline_sentiment_confidence"] < 1).mean() * 100, 1),
            ]
        },
        index=[
            "Tweets",
            "Airlines",
            "First tweet (UTC)",
            "Last tweet (UTC)",
            "Mean words per tweet",
            "Median words per tweet",
            "Mean annotator confidence",
            "% tweets with confidence < 1",
        ],
    )

    sentiment = (
        df["airline_sentiment"]
        .value_counts()
        .reindex(config.LABELS)
        .to_frame("count")
        .assign(percent=lambda t: (t["count"] / t["count"].sum() * 100).round(1))
    )

    by_airline = pd.crosstab(df["airline"], df["airline_sentiment"])
    by_airline["total"] = by_airline.sum(axis=1)
    by_airline["% negative"] = (by_airline["negative"] / by_airline["total"] * 100).round(1)
    by_airline = by_airline.sort_values("total", ascending=False)

    reasons = (
        df["negativereason"]
        .value_counts()
        .to_frame("count")
        .assign(percent=lambda t: (t["count"] / t["count"].sum() * 100).round(1))
    )
    return {
        "overview": overview,
        "sentiment": sentiment,
        "by_airline": by_airline,
        "negative_reasons": reasons,
    }
