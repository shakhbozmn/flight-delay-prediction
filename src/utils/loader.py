import pandas as pd
import streamlit as st
import joblib
from typing import Tuple, Optional, Dict, Any

@st.cache_data
def load_dataset() -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv('data/Airline_Delay_Cause.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_resource
def load_model_components() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    try:
        model = joblib.load('models/best_model.pkl')
        pipeline_data = joblib.load('models/pipeline_data.pkl')
        return model, pipeline_data
    except FileNotFoundError:
        st.error("Model files not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'total_records': df.shape[0],
        'total_features': df.shape[1],
        'time_period': f"{df['year'].min()}-{df['year'].max()}",
        'total_airlines': df['carrier'].nunique(),
        'total_airports': df['airport'].nunique(),
        'missing_values': df.isnull().sum().sum()
    }