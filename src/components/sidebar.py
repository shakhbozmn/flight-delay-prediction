import streamlit as st
from typing import Dict, Any

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown("# Flight Delay Analysis")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Overview", "Data Exploration", "Model Prediction"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("")
    
    return {
        "page": page,
        "theme": "plotly",
        "show_raw_data": False
    }