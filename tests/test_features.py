"""Tests for the deterministic feature transformer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.features import (
    BASE_INPUT_COLUMNS,
    CATEGORICAL_FEATURES,
    FeatureBuilder,
    NUMERIC_FEATURES,
)


def _single_row(month: int, arr_flights: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2024,
                "month": month,
                "carrier": "AA",
                "airport": "AAA",
                "arr_flights": arr_flights,
                "arr_cancelled": 10,
                "arr_diverted": 5,
            }
        ]
    )


def test_feature_builder_returns_all_engineered_columns() -> None:
    out = FeatureBuilder().transform(_single_row(6))
    for column in BASE_INPUT_COLUMNS:
        assert column in out.columns
    for column in NUMERIC_FEATURES:
        assert column in out.columns
    for column in CATEGORICAL_FEATURES:
        assert column in out.columns


def test_feature_builder_calendar_features_match_expectations() -> None:
    out_june = FeatureBuilder().transform(_single_row(6))
    out_december = FeatureBuilder().transform(_single_row(12))

    assert int(out_june.loc[0, "quarter"]) == 2
    assert int(out_june.loc[0, "is_winter"]) == 0
    assert int(out_june.loc[0, "is_summer"]) == 1
    assert int(out_june.loc[0, "is_peak_travel"]) == 1

    assert int(out_december.loc[0, "quarter"]) == 4
    assert int(out_december.loc[0, "is_winter"]) == 1
    assert int(out_december.loc[0, "is_summer"]) == 0
    assert int(out_december.loc[0, "is_peak_travel"]) == 1


def test_feature_builder_uses_calendar_days_in_month() -> None:
    out_feb_2024 = FeatureBuilder().transform(_single_row(2))
    out_jun_2024 = FeatureBuilder().transform(_single_row(6))
    assert out_feb_2024.loc[0, "flights_per_day"] == pytest.approx(100 / 29)
    assert out_jun_2024.loc[0, "flights_per_day"] == pytest.approx(100 / 30)


def test_feature_builder_does_not_mutate_input() -> None:
    original = _single_row(6)
    snapshot = original.copy()
    _ = FeatureBuilder().transform(original)
    pd.testing.assert_frame_equal(original, snapshot)


def test_feature_builder_rejects_invalid_month() -> None:
    bad = _single_row(13)
    with pytest.raises(ValueError, match="month outside 1\\.\\.12"):
        FeatureBuilder().transform(bad)


def test_feature_builder_rejects_negative_flights() -> None:
    bad = _single_row(6, arr_flights=-1)
    with pytest.raises(ValueError, match="negative arr_flights"):
        FeatureBuilder().transform(bad)


def test_feature_builder_rejects_disruptions_exceeding_flights() -> None:
    bad = _single_row(6)
    bad.loc[0, "arr_flights"] = 5
    bad.loc[0, "arr_cancelled"] = 4
    bad.loc[0, "arr_diverted"] = 4
    with pytest.raises(ValueError, match="cancellations\\+diversions > arr_flights"):
        FeatureBuilder().transform(bad)


def test_feature_builder_fit_returns_self() -> None:
    builder = FeatureBuilder()
    assert builder.fit(_single_row(6)) is builder