"""Tests for the BTS dataset schema and chronological split logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.data import (
    REQUIRED_COLUMNS,
    build_target,
    split_by_period,
    validate_dataset,
)


def test_required_columns_contains_mandatory_fields() -> None:
    for column in (
        "year",
        "month",
        "carrier",
        "airport",
        "arr_flights",
        "arr_del15",
        "arr_cancelled",
        "arr_diverted",
    ):
        assert column in REQUIRED_COLUMNS


def test_validate_dataset_accepts_valid_sample(bts_sample: pd.DataFrame) -> None:
    assert validate_dataset(bts_sample) is None


def test_validate_dataset_rejects_missing_columns(bts_sample: pd.DataFrame) -> None:
    broken = bts_sample.drop(columns=["arr_del15"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(broken)


def test_validate_dataset_rejects_invalid_month(bts_sample: pd.DataFrame) -> None:
    broken = bts_sample.copy()
    broken.loc[0, "month"] = 13
    with pytest.raises(ValueError, match="month values outside 1\\.\\.12"):
        validate_dataset(broken)


def test_validate_dataset_rejects_pre_2003_year(bts_sample: pd.DataFrame) -> None:
    broken = bts_sample.copy()
    broken.loc[0, "year"] = 2002
    with pytest.raises(ValueError, match="before 2003"):
        validate_dataset(broken)


def test_validate_dataset_rejects_duplicate_records(bts_sample: pd.DataFrame) -> None:
    duplicate = pd.concat([bts_sample, bts_sample.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate carrier/airport/year/month"):
        validate_dataset(duplicate)


def test_build_target_excludes_leak_columns(bts_sample: pd.DataFrame) -> None:
    features, target = build_target(bts_sample, threshold=0.25)
    assert "arr_del15" not in features.columns
    assert "arr_delay" not in features.columns
    assert "carrier_delay" not in features.columns
    assert target.tolist() == [0, 1, 0, 1, 1, 0, 1, 0]


def test_build_target_drops_zero_flight_rows() -> None:
    rows = [
        {
            "year": 2024,
            "month": 1,
            "carrier": "EE",
            "airport": "EEE",
            "arr_flights": 0,
            "arr_del15": 0,
            "arr_cancelled": 0,
            "arr_diverted": 0,
        },
        {
            "year": 2024,
            "month": 2,
            "carrier": "EE",
            "airport": "EEE",
            "arr_flights": 100,
            "arr_del15": 40,
            "arr_cancelled": 0,
            "arr_diverted": 0,
        },
    ]
    df = pd.DataFrame(rows)
    features, target = build_target(df, threshold=0.25)
    assert len(features) == 1
    assert target.tolist() == [1]


def test_split_by_period_partitions_by_year(bts_sample: pd.DataFrame) -> None:
    features, target = build_target(bts_sample, threshold=0.25)
    splits = split_by_period(features, target)
    assert set(splits.x_train["year"]) == {2022}
    assert set(splits.x_validation["year"]) == {2023}
    assert set(splits.x_test["year"]) == {2024}
    assert set(splits.x_drift["year"]) == {2025}
    assert splits.x_drift["month"].max() == 7


def test_split_by_period_rejects_empty_partition(bts_sample: pd.DataFrame) -> None:
    features, target = build_target(bts_sample, threshold=0.25)
    truncated = features[features["year"].isin({2022, 2023, 2025})].reset_index(drop=True)
    truncated_target = target.iloc[: len(truncated)].reset_index(drop=True)
    with pytest.raises(ValueError, match="empty"):
        split_by_period(truncated, truncated_target)