"""Prevent incorrect research comparisons from stale or misaligned predictions."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import SCORE_COLUMNS, significance_table, validate_predictions


def predictions():
    frame = pd.DataFrame({"tweet_id": [10, 11, 12], "y_true": [0, 1, 2], "y_pred": [0, 1, 2]})
    frame[SCORE_COLUMNS] = np.eye(3)
    return frame


def test_validation_accepts_reordered_rows():
    frame = predictions()
    validate_predictions({"a": frame, "b": frame.iloc[::-1]})


@pytest.mark.parametrize("problem", ["missing_id", "wrong_truth", "duplicate_id", "invalid_score"])
def test_validation_rejects_invalid_comparisons(problem):
    original, changed = predictions(), predictions()
    if problem == "missing_id":
        changed = changed.iloc[:-1]
    elif problem == "wrong_truth":
        changed.loc[0, "y_true"] = 1
    elif problem == "duplicate_id":
        changed.loc[0, "tweet_id"] = 11
    else:
        changed.loc[0, SCORE_COLUMNS[0]] = np.nan
    with pytest.raises(ValueError):
        validate_predictions({"a": original, "b": changed})


def test_holm_adjustment_is_reported():
    frame = predictions()
    table = significance_table({"a": frame, "b": frame.copy()}, "a", {})
    assert table.loc[0, "p_value_holm"] == 1
    assert not table.loc[0, "significant_holm_at_0.05"]
