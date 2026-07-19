"""Public machine-learning interfaces for the flight-delay portfolio project."""

from __future__ import annotations

from src.ml.data import (
    REQUIRED_COLUMNS,
    DatasetSplits,
    build_target,
    split_by_period,
    validate_dataset,
)
from src.ml.features import (
    BASE_INPUT_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureBuilder,
)

__all__ = [
    "BASE_INPUT_COLUMNS",
    "CATEGORICAL_FEATURES",
    "DatasetSplits",
    "FeatureBuilder",
    "NUMERIC_FEATURES",
    "REQUIRED_COLUMNS",
    "build_target",
    "split_by_period",
    "validate_dataset",
]