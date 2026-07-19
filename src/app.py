"""Streamlit entrypoint for the flight-delay portfolio dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.components.charts import ChartRenderer
from src.components.sidebar import render_sidebar
from src.utils.loader import (
    get_dataset_info,
    load_dashboard_dataset,
    load_model_resources,
)
from src.utils.processor import DataProcessor

st.set_page_config(
    page_title="Flight Delay Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #4f8df3 0%, #3553a8 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-success {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-warning {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main() -> None:
    df = load_dashboard_dataset()
    pipeline, metadata = load_model_resources()

    if df is None:
        st.stop()

    sidebar_config = render_sidebar()
    chart_renderer = ChartRenderer(theme=sidebar_config["theme"])
    processor = DataProcessor()

    page = sidebar_config["page"]
    if page == "Overview":
        render_overview_page(df, metadata, chart_renderer)
    elif page == "Data Exploration":
        render_exploration_page(df, processor, chart_renderer)
    elif page == "Model Prediction":
        if pipeline is None or metadata is None:
            st.error("Model resources are unavailable. Run scripts/train_model.py.")
            st.stop()
        render_prediction_page(df, pipeline, processor, metadata, chart_renderer)


def render_overview_page(df, metadata, chart_renderer) -> None:
    st.markdown(
        "<h1 class='main-header'>Flight Delay Prediction</h1>",
        unsafe_allow_html=True,
    )

    info = get_dataset_info(df)
    chart_renderer.render_metric_cards(info)

    st.markdown("## Overview")
    st.markdown(
        """
This dashboard demonstrates a leakage-safe machine-learning workflow built on
public U.S. Department of Transportation data. The model classifies each
carrier/airport/month record as either high delay (>25% of flights delayed
15+ minutes) or normal operations, using only features that would be known
ahead of time.
"""
    )

    st.markdown("## Selected model performance")
    if metadata is None:
        st.info("Model metadata unavailable.")
        return
    performance = metadata["test_metrics"]
    columns = st.columns(6)
    labels = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
    keys = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    for column, label, key in zip(columns, labels, keys):
        column.metric(label, f"{performance.get(key, 0.0):.3f}")

    st.caption(
        "Selected model: {} | Author: {} | Trained on BTS snapshot {}".format(
            metadata["selected_model"],
            metadata["author"],
            metadata["source_sha256"][:12],
        )
    )

    drift = metadata.get("drift_metrics", {})
    if drift:
        st.markdown("## Recent-period drift check")
        st.caption(
            "Separate evaluation on the January–July 2025 partial period. "
            "Treat as informational, not a final held-out test."
        )
        for key in ("accuracy", "f1", "roc_auc"):
            st.metric(f"2025 {key}", f"{drift.get(key, 0.0):.3f}")


def render_exploration_page(df, processor, chart_renderer) -> None:
    st.markdown(
        "<h1 class='main-header'>Data Exploration</h1>",
        unsafe_allow_html=True,
    )
    analysis = processor.create_analysis_dataframe(df)

    st.markdown("## Dataset overview")
    chart_renderer.render_metric_cards(get_dataset_info(df))

    st.markdown("### Raw data sample")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("## Delay distribution")
    chart_renderer.render_delay_distribution(analysis)

    st.markdown("## Temporal patterns")
    monthly = processor.get_monthly_statistics(analysis)
    chart_renderer.render_monthly_trends(monthly)

    st.markdown("## Carrier performance")
    carrier_stats, top_carriers = processor.get_carrier_statistics(analysis)
    chart_renderer.render_carrier_analysis(carrier_stats, top_carriers)

    st.markdown("## Delay causes")
    causes = processor.get_delay_causes_statistics(df)
    chart_renderer.render_delay_causes(causes)

    st.markdown("## Feature correlations")
    chart_renderer.render_correlation_heatmap(df)


def render_prediction_page(df, pipeline, processor, metadata, chart_renderer) -> None:
    st.markdown(
        "<h1 class='main-header'>Monthly Delay Risk Scenario</h1>",
        unsafe_allow_html=True,
    )

    performance = metadata["test_metrics"]
    columns = st.columns(5)
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    for column, metric in zip(columns, metrics):
        column.metric(metric.upper(), f"{performance.get(metric, 0.0):.3f}")

    st.markdown("## Scenario inputs")
    st.info(
        "Estimate the probability that a carrier/airport/month will exceed a "
        "25% delay rate. Enter the values you would forecast for that period."
    )

    with st.form("prediction_form"):
        st.markdown("### Period and carrier")
        first, second, third, fourth = st.columns(4)

        with first:
            year = st.number_input(
                "Year", min_value=2020, max_value=2030, value=2025, step=1
            )
        with second:
            month_names = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ]
            month_display = st.selectbox("Month", options=month_names, index=5)
            month = month_names.index(month_display) + 1

        with third:
            carrier_options = (
                df[["carrier", "carrier_name"]]
                .drop_duplicates()
                .sort_values("carrier_name")
            )
            carrier_display = [
                f"{row['carrier_name']} ({row['carrier']})"
                for _, row in carrier_options.iterrows()
            ]
            carrier_choice = st.selectbox(
                "Carrier", options=carrier_display, index=0
            )
            carrier = carrier_choice.split("(")[-1].rstrip(") ")

        with fourth:
            airport_options = (
                df[["airport", "airport_name"]]
                .drop_duplicates()
                .sort_values("airport_name")
            )
            airport_display = [
                f"{row['airport_name']} ({row['airport']})"
                for _, row in airport_options.iterrows()
            ]
            airport_choice = st.selectbox(
                "Airport", options=airport_display, index=0
            )
            airport = airport_choice.split("(")[-1].rstrip(") ")

        st.markdown("### Forecast operational volumes")
        ops1, ops2, ops3 = st.columns(3)
        with ops1:
            arr_flights = st.number_input(
                "Expected monthly flights",
                min_value=1,
                max_value=50000,
                value=1000,
            )
        with ops2:
            arr_cancelled = st.number_input(
                "Expected cancellations",
                min_value=0,
                max_value=int(arr_flights),
                value=min(20, int(arr_flights)),
            )
        with ops3:
            arr_diverted = st.number_input(
                "Expected diversions",
                min_value=0,
                max_value=max(0, int(arr_flights) - int(arr_cancelled)),
                value=min(5, int(arr_flights)),
            )

        submitted = st.form_submit_button(
            "Predict monthly delay risk", type="primary", use_container_width=True
        )

    if submitted:
        if arr_cancelled + arr_diverted > arr_flights:
            st.error("Cancellations + diversions cannot exceed total flights.")
            st.info("Adjust the inputs and try again.")
            return

        input_data = {
            "year": year,
            "month": month,
            "carrier": carrier,
            "airport": airport,
            "arr_flights": arr_flights,
            "arr_cancelled": arr_cancelled,
            "arr_diverted": arr_diverted,
        }

        try:
            features = processor.prepare_prediction_input(input_data)
            prediction = int(pipeline.predict(features)[0])
            scores = np.asarray(pipeline.predict_proba(features)[0])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Verify the input values and ensure the model artifact is loaded."
            )
            return

        st.markdown("## Prediction")
        chart_renderer.render_prediction_result(prediction, scores)
        st.markdown("## Historical context")
        chart_renderer.render_historical_context(df, carrier, airport)


if __name__ == "__main__":
    main()