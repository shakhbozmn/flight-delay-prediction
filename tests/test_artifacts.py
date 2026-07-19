"""Tests for the artifact metadata contract and round-trip behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from src.ml.artifacts import (
    REQUIRED_METADATA_KEYS,
    build_metadata,
    load_metadata,
    validate_metadata,
)
from src.ml.training import (
    build_candidate_pipelines,
    evaluate_candidate,
    save_model_bundle,
)


def _sample_metadata() -> dict:
    return build_metadata(
        author="Shahboz Munirov",
        source_sha256="383fb1ae404cc46aa9380bbc8156fdf6e2e4bd5af7ae1197717a639a92378134",
        target_threshold=0.25,
        partitions={
            "train": "2003-2022",
            "validation": "2023",
            "test": "2024",
            "drift": "2025-01..2025-07",
        },
        selected_model="XGBoost",
        validation_results={
            "XGBoost": {
                "accuracy": 0.79,
                "f1": 0.67,
                "precision": 0.59,
                "recall": 0.77,
                "roc_auc": 0.87,
                "pr_auc": 0.6,
            }
        },
        test_metrics={
            "accuracy": 0.78,
            "f1": 0.66,
            "precision": 0.58,
            "recall": 0.76,
            "roc_auc": 0.86,
            "pr_auc": 0.59,
        },
        drift_metrics={
            "accuracy": 0.77,
            "f1": 0.65,
            "precision": 0.57,
            "recall": 0.75,
            "roc_auc": 0.85,
            "pr_auc": 0.58,
        },
        library_versions={"python": "3.11.0", "numpy": "1.0"},
    )


def test_metadata_has_all_required_keys() -> None:
    metadata = _sample_metadata()
    for key in REQUIRED_METADATA_KEYS:
        assert key in metadata


def test_validate_metadata_rejects_unknown_schema() -> None:
    metadata = _sample_metadata()
    metadata["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        validate_metadata(metadata)


def test_validate_metadata_rejects_empty_results() -> None:
    metadata = _sample_metadata()
    metadata["validation_results"] = {}
    with pytest.raises(ValueError, match="validation_results is empty"):
        validate_metadata(metadata)


def test_save_and_load_model_bundle(tmp_path: Path) -> None:
    import pandas as pd

    from src.ml.data import (
        DatasetSplits,
        build_target,
        split_by_period,
        validate_dataset,
    )

    airports = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
    rows = [
        {
            "year": year,
            "month": month,
            "carrier": carrier,
            "carrier_name": f"Carrier {carrier}",
            "airport": airport,
            "airport_name": f"Airport {airport}",
            "arr_flights": 100,
            "arr_del15": arr_del15,
            "arr_cancelled": 1,
            "arr_diverted": 1,
            "arr_delay": 50.0,
            "carrier_delay": 10.0,
            "weather_delay": 5.0,
            "nas_delay": 8.0,
            "security_delay": 0.0,
            "late_aircraft_delay": 6.0,
        }
        for year in (2022, 2023, 2024)
        for month in (1, 2, 3, 4)
        for carrier in ("AA", "BB")
        for airport in airports
        for arr_del15 in (10, 30)
    ]
    df = pd.DataFrame(rows)
    validate_dataset(df)
    features, target = build_target(df, threshold=0.25)
    splits = split_by_period(features, target)

    pipelines = build_candidate_pipelines(splits.y_train, random_state=42)
    pipeline = pipelines["LogisticRegression"]
    bundle = evaluate_candidate(pipeline, splits.x_train, splits.y_train)

    metadata = _sample_metadata()
    paths = save_model_bundle(pipeline, metadata, tmp_path)
    assert paths["pipeline_path"].exists()
    assert paths["metadata_path"].exists()

    loaded = joblib.load(paths["pipeline_path"])
    sample = splits.x_train.iloc[[0]]
    assert loaded.predict(sample).tolist() == pipeline.predict(sample).tolist()

    loaded_metadata = load_metadata(paths["metadata_path"])
    assert loaded_metadata["selected_model"] == metadata["selected_model"]
    assert json.loads(paths["metadata_path"].read_text()) == metadata
    assert set(bundle.keys()) == set(
        ("accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc")
    )