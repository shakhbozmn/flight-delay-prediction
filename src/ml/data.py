"""Dataset schema validation, target construction, and chronological splits.

The portfolio project only uses monthly aggregated BTS records. Every row is a
single carrier/airport/month observation. Source snapshots must satisfy a
strict column contract and the chronological partitions enforced by
``split_by_period``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "carrier",
    "carrier_name",
    "airport",
    "airport_name",
    "arr_flights",
    "arr_del15",
    "arr_cancelled",
    "arr_diverted",
    "arr_delay",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
)

TRAIN_YEARS: tuple[int, ...] = tuple(range(2003, 2023))
VALIDATION_YEAR: int = 2023
TEST_YEAR: int = 2024
DRIFT_YEAR: int = 2025
DRIFT_LAST_INCLUSIVE_MONTH: int = 7


@dataclass(frozen=True)
class DatasetSplits:
    """Chronological partitions with paired features and target series."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    x_drift: pd.DataFrame
    y_drift: pd.Series


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate a raw BTS snapshot.

    Raises:
        ValueError: when required columns are missing, year/month values fall
            outside supported ranges, or duplicate carrier/airport/year/month
            records are present.
    """

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "BTS snapshot is missing required columns: " + ", ".join(missing)
        )

    if df["month"].isna().any():
        raise ValueError("BTS snapshot contains null month values")
    if (df["month"] < 1).any() or (df["month"] > 12).any():
        raise ValueError("BTS snapshot contains month values outside 1..12")

    if df["year"].isna().any():
        raise ValueError("BTS snapshot contains null year values")
    if (df["year"] < 2003).any():
        raise ValueError("BTS snapshot contains year values before 2003")
    if (df["year"] > DRIFT_YEAR).any():
        raise ValueError(
            f"BTS snapshot contains year values after {DRIFT_YEAR}"
        )

    duplicates = df.duplicated(
        subset=["carrier", "airport", "year", "month", "arr_flights", "arr_del15"]
    )
    if duplicates.any():
        raise ValueError(
            "BTS snapshot contains duplicate carrier/airport/year/month records"
        )


def build_target(
    df: pd.DataFrame, threshold: float = 0.25
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a feature frame and binary delay target.

    The target equals one whenever the carrier/airport/month record has more
    than ``threshold`` of flights delayed by 15+ minutes. Rows missing the
    fields needed to compute the target are dropped.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in the open interval (0, 1)")

    df = df.copy()

    mask = (
        df["arr_flights"].notna()
        & df["arr_del15"].notna()
        & (df["arr_flights"] > 0)
    )

    work = df.loc[mask].copy()
    rate = work["arr_del15"] / work["arr_flights"].replace(0, 1)
    target = (rate > threshold).astype(int)
    target.name = "high_delay_month"

    feature_columns = [
        "year",
        "month",
        "carrier",
        "airport",
        "arr_flights",
        "arr_cancelled",
        "arr_diverted",
    ]

    features = work.loc[:, feature_columns].reset_index(drop=True)
    target = target.reset_index(drop=True)

    return features, target


def split_by_period(
    features: pd.DataFrame, target: pd.Series
) -> DatasetSplits:
    """Partition features and target by year ranges."""

    if len(features) != len(target):
        raise ValueError("features and target must have equal length")

    def _select(
        years: tuple[int, ...],
        *,
        drift_filter: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        mask = features["year"].isin(years)
        if drift_filter:
            mask = mask & (features["month"] <= DRIFT_LAST_INCLUSIVE_MONTH)
        subset_x = features.loc[mask].reset_index(drop=True)
        subset_y = target.loc[mask].reset_index(drop=True)
        return subset_x, subset_y

    x_train, y_train = _select(TRAIN_YEARS)
    x_validation, y_validation = _select((VALIDATION_YEAR,))
    x_test, y_test = _select((TEST_YEAR,))
    x_drift, y_drift = _select((DRIFT_YEAR,), drift_filter=True)

    if any(frame.empty for frame in (x_train, x_validation, x_test)):
        raise ValueError(
            "One of train/validation/test partitions is empty for the "
            "configured year boundaries"
        )

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        x_drift=x_drift,
        y_drift=y_drift,
    )