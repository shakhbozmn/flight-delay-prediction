"""Leakage-safe model pipelines and selection workflow."""

from __future__ import annotations

import platform
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.ml.data import DatasetSplits
from src.ml.evaluation import (
    REQUIRED_METRIC_KEYS,
    metric_bundle,
    select_best_validation_result,
)
from src.ml.features import (
    BASE_INPUT_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureBuilder,
)

NUMERIC_INPUT_FEATURES: tuple[str, ...] = (
    "arr_flights",
    "arr_cancelled",
    "arr_diverted",
)

MODEL_NAMES: tuple[str, ...] = (
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_INPUT_FEATURES)
                + [
                    name
                    for name in NUMERIC_FEATURES
                    if name not in NUMERIC_INPUT_FEATURES
                ],
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def _candidate_pipelines(
    y_train: pd.Series, random_state: int
) -> dict[str, Pipeline]:
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = (negative / positive) if positive > 0 else 1.0

    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("features", FeatureBuilder()),
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("features", FeatureBuilder()),
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("features", FeatureBuilder()),
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def build_candidate_pipelines(
    y_train: pd.Series, random_state: int = 42
) -> dict[str, Pipeline]:
    """Return a fresh dict of sklearn pipelines keyed by display name."""

    return _candidate_pipelines(y_train, random_state)


def evaluate_candidate(
    pipeline: Pipeline,
    x_frame: pd.DataFrame,
    y_series: pd.Series,
) -> dict[str, float]:
    """Fit and evaluate one pipeline. Used for fixtures and training."""

    pipeline.fit(x_frame, y_series)
    predictions = pipeline.predict(x_frame)
    scores = pipeline.predict_proba(x_frame)[:, 1]
    return metric_bundle(np.asarray(y_series), predictions, scores)


def validate_metric_bundle(bundle: Mapping[str, float]) -> None:
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in bundle]
    if missing:
        raise ValueError(
            "metric bundle missing required keys: " + ", ".join(missing)
        )


def library_versions() -> dict[str, str]:
    """Capture runtime library versions for artifact metadata."""

    import numpy
    import pandas
    import sklearn
    import xgboost
    import joblib

    versions: dict[str, str] = {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
        "python": platform.python_version(),
    }
    return versions


def run_validation(
    splits: DatasetSplits,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Fit each candidate on the training partition and score validation."""

    pipelines = _candidate_pipelines(splits.y_train, random_state)
    validation_results: dict[str, dict[str, float]] = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(splits.x_train, splits.y_train)
        predictions = pipeline.predict(splits.x_validation)
        scores = pipeline.predict_proba(splits.x_validation)[:, 1]
        validation_results[name] = metric_bundle(
            np.asarray(splits.y_validation),
            predictions,
            scores,
        )
        validate_metric_bundle(validation_results[name])
    return validation_results


def train_and_evaluate(
    splits: DatasetSplits,
    output_dir,
    random_state: int = 42,
) -> dict:
    """Run selection and final fit. Returns metadata for artifact export."""

    output_dir = Path_or_str(output_dir)
    validation_results = run_validation(splits, random_state)
    selected_name = select_best_validation_result(validation_results)

    candidate_pipelines = _candidate_pipelines(splits.y_train, random_state)
    selected_pipeline = candidate_pipelines[selected_name]

    fit_frame = pd.concat([splits.x_train, splits.x_validation], ignore_index=True)
    fit_target = pd.concat([splits.y_train, splits.y_validation], ignore_index=True)
    selected_pipeline.fit(fit_frame, fit_target)

    test_predictions = selected_pipeline.predict(splits.x_test)
    test_scores = selected_pipeline.predict_proba(splits.x_test)[:, 1]
    test_metrics = metric_bundle(
        np.asarray(splits.y_test), test_predictions, test_scores
    )
    validate_metric_bundle(test_metrics)

    drift_metrics: dict[str, float] = {}
    if not splits.x_drift.empty:
        drift_predictions = selected_pipeline.predict(splits.x_drift)
        drift_scores = selected_pipeline.predict_proba(splits.x_drift)[:, 1]
        drift_metrics = metric_bundle(
            np.asarray(splits.y_drift), drift_predictions, drift_scores
        )

    return {
        "pipeline": selected_pipeline,
        "selected_model": selected_name,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "drift_metrics": drift_metrics,
    }


def Path_or_str(value):  # pragma: no cover - tiny helper
    from pathlib import Path

    return Path(value) if not isinstance(value, Path) else value


def save_model_bundle(pipeline, metadata, output_dir) -> dict[str, object]:
    """Persist a fitted pipeline and metadata JSON, returning file paths."""

    output_dir = Path_or_str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_dir / "model_pipeline.joblib"
    metadata_path = output_dir / "model_metadata.json"

    joblib.dump(pipeline, pipeline_path)

    import json

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {"pipeline_path": pipeline_path, "metadata_path": metadata_path}