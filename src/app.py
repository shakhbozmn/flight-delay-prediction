import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from utils.loader import load_dataset, load_model_components, get_dataset_info
from utils.processor import DataProcessor
from components.sidebar import render_sidebar
from components.charts import ChartRenderer

st.set_page_config(
    page_title="Flight Delay Analysis & Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
""", unsafe_allow_html=True)

def main():
    
    df = load_dataset()
    model, pipeline_data = load_model_components()
    
    if df is None:
        st.stop()
    
    if model and pipeline_data:
        processor = DataProcessor(pipeline_data)
    
    sidebar_config = render_sidebar()
    
    chart_renderer = ChartRenderer(theme=sidebar_config["theme"])
    
    if sidebar_config["page"] == "Overview":
        render_overview_page(df, sidebar_config, chart_renderer)
    elif sidebar_config["page"] == "Data Exploration":
        render_exploration_page(df, processor, sidebar_config, chart_renderer)
    elif sidebar_config["page"] == "Model Prediction":
        if model and pipeline_data:
            render_prediction_page(df, model, processor, sidebar_config, chart_renderer)
        else:
            st.error("Model components not loaded. Please check model files.")

def render_overview_page(df, sidebar_config, chart_renderer):
    st.markdown('<h1 class="main-header">Flight Delay Analysis & Prediction System</h1>', unsafe_allow_html=True)
    
    dataset_info = get_dataset_info(df)
    chart_renderer.render_metric_cards(dataset_info)
    
    st.markdown("---")
    
    st.markdown("## Overview")
    st.markdown("""
    This work presents a comprehensive machine learning analysis of airline delays using the U.S. Department of Transportation's Airline On-Time Statistics and Delay Causes dataset. The project develops a complete data science pipeline to predict high-delay periods and extract actionable insights for airline operations.
    
    ### Objectives:
    1. **Predictive Modeling**: Develop machine learning models to classify months with high delay rates (>25%) vs. normal operations
    2. **Operational Insights**: Identify key factors contributing to flight delays through exploratory data analysis
    3. **Performance Comparison**: Evaluate multiple algorithms to determine the most effective approach for delay prediction
    4. **Feature Engineering**: Create meaningful predictors from raw operational data to improve model performance
    """)
    
    st.markdown("---")
    st.markdown("## Best Model Performance")
    
    st.success("""
    **Best Model**: XGBoost  
    **F1 Score**: 0.667  
    **Accuracy**: 79.3%  
    **ROC-AUC**: 0.873
    """)

def render_exploration_page(df, processor, sidebar_config, chart_renderer):
    st.markdown('<h1 class="main-header">Data Exploration Dashboard</h1>', unsafe_allow_html=True)
    
    df_analysis = processor.create_analysis_dataframe(df)
    
    st.markdown("## Dataset Overview")
    dataset_info = get_dataset_info(df)
    chart_renderer.render_metric_cards(dataset_info)
    

    st.markdown("### Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## Delay Distribution Analysis")
    chart_renderer.render_delay_distribution(df_analysis)
    
    st.markdown("---")
    
    st.markdown("## Temporal Patterns")
    monthly_stats = processor.get_monthly_statistics(df_analysis)
    chart_renderer.render_monthly_trends(monthly_stats)
    
    st.markdown("---")
    
    st.markdown("## Airline Performance Analysis")
    carrier_stats, top_carriers = processor.get_carrier_statistics(df_analysis)
    chart_renderer.render_carrier_analysis(carrier_stats, top_carriers)
    
    st.markdown("---")
    
    st.markdown("## Delay Causes Analysis")
    delay_means = processor.get_delay_causes_statistics(df)
    chart_renderer.render_delay_causes(delay_means)
    
    st.markdown("---")
    
    st.markdown("## Feature Correlations")
    chart_renderer.render_correlation_heatmap(df)

def render_prediction_page(df, model, processor, sidebar_config, chart_renderer):
    st.markdown('<h1 class="main-header">Monthly Delay Risk Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("## Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", processor.pipeline_data['model_name'])
    with col2:
        st.metric("F1-Score", f"{processor.pipeline_data['model_performance']['f1_score']:.4f}")
    with col3:
        st.metric("Accuracy", f"{processor.pipeline_data['model_performance']['accuracy']:.4f}")
    with col4:
        st.metric("ROC-AUC", f"{processor.pipeline_data['model_performance']['roc_auc']:.4f}")
    
    st.markdown("---")
    
    st.markdown("## Prediction Input")
    st.info("**Objective**: Predict if a specific month will have high delay rates (>25% of flights delayed) for a carrier-airport combination")
    
    with st.form("prediction_form"):
        st.markdown("### Basic Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            year = st.number_input("Year", min_value=2020, max_value=2035, value=2025, step=1)
        
        with col2:
            month_names = ["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"]
            month_display = st.selectbox("Month", options=month_names, index=5)
            month = month_names.index(month_display) + 1
        
        with col3:
            carrier_options = df[['carrier', 'carrier_name']].drop_duplicates().sort_values('carrier_name')
            carrier_display_options = [f"{row['carrier_name']} ({row['carrier']})" for _, row in carrier_options.iterrows()]
            carrier_selection = st.selectbox("Airline Carrier", options=carrier_display_options, index=0)
            carrier = carrier_selection.split('(')[-1].replace(')', '')

        with col4:
            airport_options = df[['airport', 'airport_name']].drop_duplicates().sort_values('airport_name')
            airport_display_options = [f"{row['airport_name']} ({row['airport']})" for _, row in airport_options.iterrows()]
            airport_selection = st.selectbox("Airport", options=airport_display_options, index=0)
            airport = airport_selection.split('(')[-1].replace(')', '')
        
        st.markdown("### Monthly Operations Forecast")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            arr_flights = st.number_input("Expected Monthly Flights", 
                                        min_value=1, max_value=50000, value=1000,
                                        help="Total flights expected for this carrier-airport combination in the month")
        
        with col2:
            arr_cancelled = st.number_input("Expected Cancellations", 
                                          min_value=0, max_value=int(arr_flights), value=min(20, int(arr_flights)),
                                          help="Expected cancelled flights for the month")
        
        with col3:
            arr_diverted = st.number_input("Expected Diversions", 
                                         min_value=0, max_value=int(arr_flights), value=min(5, int(arr_flights)),
                                         help="Expected diverted flights for the month")
            
        # Calculate metrics for display
        total_disruptions = arr_cancelled + arr_diverted
        
        st.markdown("### Calculated Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        quarter = ((month - 1) // 3) + 1
        is_winter = month in [12, 1, 2]
        is_summer = month in [6, 7, 8]
        is_peak_travel = month in [6, 7, 8, 11, 12]
        flights_per_day = arr_flights / 30
        cancellation_rate = arr_cancelled / arr_flights if arr_flights > 0 else 0
        total_disruptions = arr_cancelled + arr_diverted
        
        with col1:
            st.metric("Quarter", quarter)
            st.metric("Flights/Day", f"{flights_per_day:.1f}")
        
        with col2:
            st.metric("Winter Season", "Yes" if is_winter else "No")
            st.metric("Cancellation Rate", f"{cancellation_rate:.3f}")
        
        with col3:
            st.metric("Summer Season", "Yes" if is_summer else "No")
            st.metric("Total Disruptions", total_disruptions)
        
        with col4:
            st.metric("Peak Travel", "Yes" if is_peak_travel else "No")
            
        st.markdown("---")
        submitted = st.form_submit_button("Predict Monthly Delay Risk", type="primary", use_container_width=True)
    
    if submitted:
        # Validate inputs after form submission
        total_disruptions = arr_cancelled + arr_diverted
        is_valid = total_disruptions <= arr_flights
        
        if not is_valid:
            st.error(f"❌ Invalid input: Cancellations ({arr_cancelled}) + Diversions ({arr_diverted}) = {total_disruptions} cannot exceed total flights ({arr_flights})")
            st.info("Please adjust the values and try again.")
        else:
            try:
                input_data = {
                    'year': year, 'month': month, 'carrier': carrier, 'airport': airport,
                    'arr_flights': arr_flights, 'arr_cancelled': arr_cancelled, 'arr_diverted': arr_diverted
                }
                
                processed_data = processor.prepare_prediction_input(input_data)
                prediction = model.predict(processed_data)[0]
                prediction_proba = model.predict_proba(processed_data)[0]
                
                st.markdown("---")
                st.markdown("## Prediction Results")
                chart_renderer.render_prediction_result(prediction, prediction_proba)
                
                st.markdown("---")
                st.markdown("## Historical Context")
                chart_renderer.render_historical_context(df, carrier, airport)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()