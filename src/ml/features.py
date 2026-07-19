"""Deterministic feature engineering that learns no statistics in ``fit``.

The transformer is safe to call before the train/test split; it only depends
on the calendar and the values present in the row. Categorical columns are
left as-is for downstream encoders to handle so that pipeline preprocessing
remains a learning step.
"""

from __future__ import annotations

import calendar

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_INPUT_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "carrier",
    "airport",
    "arr_flights",
    "arr_cancelled",
    "arr_diverted",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "arr_flights",
    "arr_cancelled",
    "arr_diverted",
    "flights_per_day",
    "cancellation_rate",
    "total_disruptions",
    "quarter",
    "is_winter",
    "is_summer",
    "is_peak_travel",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("carrier", "airport")

WINTER_MONTHS: frozenset[int] = frozenset({12, 1, 2})
SUMMER_MONTHS: frozenset[int] = frozenset({6, 7, 8})
PEAK_TRAVEL_MONTHS: frozenset[int] = frozenset({6, 7, 8, 11, 12})


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Add deterministic calendar and operational features."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "FeatureBuilder":
        """Return ``self``. No statistics are learned."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a new frame with engineered columns appended."""

        work = self._validate(X).copy()
        work["quarter"] = ((work["month"] - 1) // 3) + 1
        work["is_winter"] = work["month"].isin(WINTER_MONTHS).astype(int)
        work["is_summer"] = work["month"].isin(SUMMER_MONTHS).astype(int)
        work["is_peak_travel"] = work["month"].isin(PEAK_TRAVEL_MONTHS).astype(int)

        days_in_month = work.apply(
            lambda row: calendar.monthrange(int(row["year"]), int(row["month"]))[1],
            axis=1,
        )
        work["flights_per_day"] = work["arr_flights"] / days_in_month
        work["cancellation_rate"] = (
            work["arr_cancelled"] / work["arr_flights"].replace(0, 1)
        )
        work["total_disruptions"] = work["arr_cancelled"] + work["arr_diverted"]
        return work

    @staticmethod
    def _validate(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureBuilder expects a pandas DataFrame input")
        missing = [column for column in BASE_INPUT_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(
                "FeatureBuilder missing required input columns: "
                + ", ".join(missing)
            )

        if (X["month"] < 1).any() or (X["month"] > 12).any():
            raise ValueError("FeatureBuilder received month outside 1..12")
        if (X["arr_flights"] < 0).any():
            raise ValueError("FeatureBuilder received negative arr_flights")
        if ((X["arr_cancelled"] + X["arr_diverted"]) > X["arr_flights"]).any():
            raise ValueError(
                "FeatureBuilder received cancellations+diversions > arr_flights"
            )
        return X


def engineered_feature_columns() -> list[str]:
    """Return the engineered numeric column names in stable order."""

    return list(NUMERIC_FEATURES)