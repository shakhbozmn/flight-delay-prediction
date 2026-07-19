"""Streamlit sidebar with personal branding and navigation."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

GITHUB_PROFILE_URL = "https://github.com/shakhbozmn"


def render_sidebar() -> Dict[str, Any]:
    """Render the portfolio sidebar and return its navigation config."""

    st.sidebar.markdown("# Flight Delay Prediction")
    st.sidebar.markdown(
        "Personal portfolio project analysing U.S. DOT flight delays."
    )
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Overview", "Data Exploration", "Model Prediction"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Author:** [Shahboz Munirov]({})".format(GITHUB_PROFILE_URL)
    )
    st.sidebar.markdown(
        "**Scope:** Educational scenario estimator based on historical BTS data."
    )

    return {"page": page, "theme": "plotly", "show_raw_data": False}