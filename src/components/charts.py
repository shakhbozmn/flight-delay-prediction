"""Chart helpers used by the Streamlit app."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st


class ChartRenderer:
    """Render the dashboard's Plotly and matplotlib visualizations."""

    def __init__(self, theme: str = "plotly") -> None:
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set2

    def render_metric_cards(self, metrics: dict) -> None:
        columns = st.columns(4)
        labels = [
            ("Total records", f"{metrics['total_records']:,}"),
            ("Airlines", f"{metrics['total_airlines']:,}"),
            ("Airports", f"{metrics['total_airports']:,}"),
            ("Time period", metrics["time_period"]),
        ]
        for column, (label, value) in zip(columns, labels):
            column.metric(label, value)

    def render_delay_distribution(self, df: pd.DataFrame) -> None:
        first, second = st.columns(2)
        with first:
            fig = px.histogram(
                df,
                x="arr_delay",
                nbins=50,
                title="Arrival delay distribution",
                template=self.theme,
                color_discrete_sequence=self.color_palette,
            )
            fig.update_layout(
                xaxis_title="Delay minutes", yaxis_title="Count", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        with second:
            fig = px.histogram(
                df,
                x="delay_rate",
                nbins=50,
                title="Delay rate distribution",
                template=self.theme,
                color_discrete_sequence=self.color_palette,
            )
            fig.update_layout(
                xaxis_title="Delay rate", yaxis_title="Count", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_monthly_trends(self, monthly_stats: pd.DataFrame) -> None:
        first, second = st.columns(2)
        with first:
            fig = px.bar(
                monthly_stats,
                x="month",
                y="arr_delay",
                title="Average arrival delay by month",
                template=self.theme,
                color="arr_delay",
                color_continuous_scale="Reds",
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Average delay (minutes)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with second:
            fig = px.line(
                monthly_stats,
                x="month",
                y="delay_rate",
                title="Delay rate trend by month",
                markers=True,
                template=self.theme,
            )
            fig.update_traces(line_color="#FF6B6B", marker_color="#FF6B6B")
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Delay rate",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_carrier_analysis(
        self, carrier_stats: pd.DataFrame, top_carriers: pd.Series
    ) -> None:
        first, second = st.columns(2)
        with first:
            fig = px.bar(
                x=top_carriers.index,
                y=top_carriers.values,
                title="Flight volume by carrier (top 10)",
                template=self.theme,
                color=top_carriers.values,
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                xaxis_title="Carrier",
                yaxis_title="Total flights",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with second:
            fig = px.bar(
                x=carrier_stats.index,
                y=carrier_stats["arr_delay"],
                title="Average arrival delay by carrier (top 10)",
                template=self.theme,
                color=carrier_stats["arr_delay"],
                color_continuous_scale="Oranges",
            )
            fig.update_layout(
                xaxis_title="Carrier",
                yaxis_title="Average delay (minutes)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_delay_causes(self, delay_means: pd.Series) -> None:
        first, second = st.columns(2)
        with first:
            fig = px.bar(
                x=delay_means.index,
                y=delay_means.values,
                title="Average delay by cause",
                template=self.theme,
                color=delay_means.values,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                xaxis_title="Delay cause",
                yaxis_title="Average minutes",
                showlegend=False,
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        with second:
            fig = px.pie(
                values=delay_means.values,
                names=delay_means.index,
                title="Delay cause distribution",
                template=self.theme,
                color_discrete_sequence=self.color_palette,
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_correlation_heatmap(self, df: pd.DataFrame) -> None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature correlation matrix", fontsize=16, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    def render_prediction_result(
        self, prediction: int, prediction_proba: np.ndarray
    ) -> None:
        confidence = float(
            prediction_proba[1] if prediction == 1 else prediction_proba[0]
        )
        if prediction == 1:
            st.markdown(
                f"<div class='prediction-warning'>Predicted high delay risk "
                f"(confidence {confidence:.2f})</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='prediction-success'>Predicted normal operations "
                f"(confidence {confidence:.2f})</div>",
                unsafe_allow_html=True,
            )

    def render_historical_context(
        self, df: pd.DataFrame, carrier: str, airport: str
    ) -> None:
        history = (
            df.loc[(df["carrier"] == carrier) & (df["airport"] == airport)]
            .sort_values(["year", "month"])
        )
        if history.empty:
            st.info(
                "No historical context available for this carrier/airport combination."
            )
            return
        fig = px.line(
            history,
            x="month",
            y="arr_delay",
            color="year",
            title=f"Historical arrival delay for {carrier} at {airport}",
            template=self.theme,
        )
        st.plotly_chart(fig, use_container_width=True)