"""Tests for selection, metric bundles, and candidate pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.data import (
    DatasetSplits,
    build_target,
    split_by_period,
    validate_dataset,
)
from src.ml.evaluation import (
    REQUIRED_METRIC_KEYS,
    metric_bundle,
    select_best_validation_result,
)
from src.ml.training import (
    build_candidate_pipelines,
    evaluate_candidate,
)


def _splits() -> DatasetSplits:
    rows = []
    for year in (2022, 2023, 2024, 2025):
        months = range(1, 4) if year == 2025 else range(1, 13)
        for month in months:
            if year == 2025 and month > 7:
                continue
            for carrier in ("AA", "BB"):
                for airport in ("AAA", "BBB"):
                    arr_flights = 100
                    arr_del15 = 20 if month % 2 == 0 else 30
                    rows.append(
                        {
                            "year": year,
                            "month": month,
                            "carrier": carrier,
                            "carrier_name": f"Carrier {carrier}",
                            "airport": airport,
                            "airport_name": f"Airport {airport}",
                            "arr_flights": arr_flights,
                            "arr_del15": arr_del15,
                            "arr_cancelled": 5,
                            "arr_diverted": 1,
                            "arr_delay": 100.0,
                            "carrier_delay": 30.0,
                            "weather_delay": 10.0,
                            "nas_delay": 20.0,
                            "security_delay": 0.0,
                            "late_aircraft_delay": 15.0,
                        }
                    )
    df = pd.DataFrame(rows)
    validate_dataset(df)
    features, target = build_target(df, threshold=0.25)
    return split_by_period(features, target)


def test_metric_bundle_contains_required_keys() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    bundle = metric_bundle(y_true, y_pred, y_score)
    assert set(bundle.keys()) == set(REQUIRED_METRIC_KEYS)
    assert bundle["accuracy"] == 1.0
    assert bundle["f1"] == 1.0


def test_metric_bundle_requires_equal_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        metric_bundle(np.array([0, 1]), np.array([0]), np.array([0.1, 0.2]))


def test_select_best_validation_result_prefers_f1() -> None:
    results = {
        "alpha": {"f1": 0.5, "pr_auc": 0.4, "roc_auc": 0.6},
        "beta": {"f1": 0.7, "pr_auc": 0.5, "roc_auc": 0.7},
    }
    assert select_best_validation_result(results) == "beta"


def test_select_best_validation_result_breaks_ties_by_name() -> None:
    results = {
        "alpha": {"f1": 0.5, "pr_auc": 0.5, "roc_auc": 0.7},
        "beta": {"f1": 0.5, "pr_auc": 0.5, "roc_auc": 0.7},
    }
    assert select_best_validation_result(results) == "beta"


def test_select_best_validation_result_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_best_validation_result({})


def test_build_candidate_pipelines_returns_three_models() -> None:
    splits = _splits()
    pipelines = build_candidate_pipelines(splits.y_train, random_state=42)
    assert set(pipelines.keys()) == {
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
    }


def test_candidate_pipeline_handles_unknown_categories() -> None:
    splits = _splits()
    pipelines = build_candidate_pipelines(splits.y_train, random_state=42)
    pipeline = pipelines["LogisticRegression"]
    pipeline.fit(splits.x_train, splits.y_train)

    unknown_row = pd.DataFrame(
        [
            {
                "year": 2023,
                "month": 6,
                "carrier": "ZZ",
                "airport": "ZZZ",
                "arr_flights": 100,
                "arr_cancelled": 5,
                "arr_diverted": 1,
            }
        ]
    )
    predictions = pipeline.predict(unknown_row)
    assert predictions.shape == (1,)


def test_evaluate_candidate_returns_metric_bundle() -> None:
    splits = _splits()
    pipelines = build_candidate_pipelines(splits.y_train, random_state=42)
    pipeline = pipelines["LogisticRegression"]
    bundle = evaluate_candidate(pipeline, splits.x_train, splits.y_train)
    assert set(bundle.keys()) == set(REQUIRED_METRIC_KEYS)