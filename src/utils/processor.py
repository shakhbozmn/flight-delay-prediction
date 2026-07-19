"""EDA helpers and scenario-row construction for the Streamlit app."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ml.features import BASE_INPUT_COLUMNS


class DataProcessor:
    """Provide EDA-style aggregations and prepare prediction input rows.

    The class no longer depends on model artifacts. All training-time
    statistics live inside the persisted pipeline; this processor only
    performs deterministic dataframe transformations needed by the UI.
    """

    def create_analysis_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        analysis = df.copy()
        analysis["delay_rate"] = analysis["arr_del15"] / analysis[
            "arr_flights"
        ].replace(0, 1)
        return analysis.dropna(subset=["delay_rate"])

    def get_monthly_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        monthly = (
            df.groupby("month")
            .agg(
                {
                    "arr_delay": "mean",
                    "delay_rate": "mean",
                    "arr_cancelled": "mean",
                    "arr_diverted": "mean",
                }
            )
            .round(3)
            .reset_index()
        )
        return monthly

    def get_carrier_statistics(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> tuple[pd.DataFrame, pd.Series]:
        top_carriers = df["carrier"].value_counts().head(top_n)
        carrier_stats = (
            df.groupby("carrier")
            .agg(
                {
                    "arr_delay": "mean",
                    "delay_rate": "mean",
                    "arr_flights": "sum",
                }
            )
            .round(3)
        )
        return carrier_stats.loc[top_carriers.index], top_carriers

    def get_delay_causes_statistics(self, df: pd.DataFrame) -> pd.Series:
        causes = [
            "carrier_delay",
            "weather_delay",
            "nas_delay",
            "security_delay",
            "late_aircraft_delay",
        ]
        return df[causes].mean().round(2)

    def prepare_prediction_input(self, input_data: dict[str, Any]) -> pd.DataFrame:
        """Return one-row scenario frame with exactly ``BASE_INPUT_COLUMNS``."""

        row = {column: input_data[column] for column in BASE_INPUT_COLUMNS}
        return pd.DataFrame([row])