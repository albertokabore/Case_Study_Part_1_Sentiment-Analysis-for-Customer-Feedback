"""Verify saved research results against the existing Tweets.csv; add no datasets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src import config
from src.classical_models import get_model_specs
from src.data_loading import clean_dataset, load_raw, split_data, summary_statistics
from src.evaluation import SCORE_COLUMNS, compute_metrics, load_predictions, validate_predictions


def main() -> None:
    data, cleaning = clean_dataset(load_raw())
    splits = split_data(data)
    predictions = load_predictions()
    validate_predictions(predictions, splits.test)
    comparison = pd.read_csv(config.METRICS_DIR / "model_comparison.csv").set_index("slug")
    assert set(comparison.index) == set(predictions), "Comparison omits or adds models."
    errors = {}
    for slug, frame in predictions.items():
        computed = compute_metrics(frame.y_true, frame.y_pred, frame[SCORE_COLUMNS])
        errors[slug] = max(abs(value - comparison.loc[slug, metric]) for metric, value in computed.items())
        assert errors[slug] < 1e-10, f"{slug}: saved metrics differ from predictions."
    assignment = pd.read_csv(config.METRICS_DIR / "split_assignment.csv").set_index("tweet_id")["split"]
    assert assignment.index.is_unique and set(assignment.index) == set(data.tweet_id)
    for part in ("train", "val", "test"):
        assert set(assignment[assignment == part].index) == set(getattr(splits, part).tweet_id)
    assert cleaning == json.loads((config.METRICS_DIR / "cleaning_report.json").read_text())
    actual_sentiment = summary_statistics(data)["sentiment"]
    saved_sentiment = pd.read_csv(config.METRICS_DIR / "summary_sentiment.csv", index_col=0)
    assert np.array_equal(actual_sentiment.values, saved_sentiment.loc[actual_sentiment.index].values)
    provenance = {}
    for spec in get_model_specs():
        info = json.loads((config.METRICS_DIR / "models" / f"{spec.slug}.json").read_text())
        provenance[spec.slug] = {"saved_candidates": info["n_candidates"],
                                 "current_grid_candidates": len(ParameterGrid(spec.param_grid)),
                                 "historical_grid_saved": "param_grid" in info}
    result = {
        "cleaning": cleaning,
        "split_sizes": {p: len(getattr(splits, p)) for p in ("train", "val", "test")},
        "validated_models": sorted(predictions),
        "maximum_metric_absolute_errors": errors,
        "test_development_shared_lowercased_texts": len(set(splits.test.text.str.lower()) & set(splits.train_full.text.str.lower())),
        "search_provenance": provenance,
        "transformer_results_available": "distilroberta" in predictions,
    }
    output = config.METRICS_DIR / "research_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
