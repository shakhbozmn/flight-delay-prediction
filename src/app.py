import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Delay Analysis & Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models
@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('data/Airline_Delay_Cause.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'data/Airline_Delay_Cause.csv' exists.")
        return None

@st.cache_resource
def load_model_and_components():
    """Load the trained model and preprocessing components"""
    try:
        # Load model
        model = joblib.load('models/best_model.pkl')
        
        # Load preprocessing components
        pipeline_data = joblib.load('models/pipeline_data.pkl')
        
        return model, pipeline_data
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model files exist in 'models/' directory.")
        return None, None

def prepare_input_for_prediction(input_data, pipeline_data):
    """Prepare input data for model prediction"""
    # Get components from pipeline
    label_encoders = pipeline_data['label_encoders']
    scaler = pipeline_data['scaler']
    feature_columns = pipeline_data['feature_columns']
    airport_stats = pipeline_data['airport_stats']
    carrier_stats = pipeline_data['carrier_stats']
    
    # Create feature dictionary with all required features
    feature_dict = {}
    
    # Basic features
    feature_dict['year'] = input_data['year']
    feature_dict['month'] = input_data['month']
    feature_dict['arr_flights'] = input_data['arr_flights']
    feature_dict['arr_cancelled'] = input_data['arr_cancelled']
    feature_dict['arr_diverted'] = input_data['arr_diverted']
    
    # Time-based features
    feature_dict['quarter'] = ((input_data['month'] - 1) // 3) + 1
    feature_dict['is_winter'] = 1 if input_data['month'] in [12, 1, 2] else 0
    feature_dict['is_summer'] = 1 if input_data['month'] in [6, 7, 8] else 0
    feature_dict['is_peak_travel'] = 1 if input_data['month'] in [6, 7, 8, 11, 12] else 0
    
    # Operational features
    feature_dict['flights_per_day'] = input_data['arr_flights'] / 30
    feature_dict['cancellation_rate'] = input_data['arr_cancelled'] / input_data['arr_flights'] if input_data['arr_flights'] > 0 else 0
    feature_dict['total_disruptions'] = input_data['arr_cancelled'] + input_data['arr_diverted']
    
    # Airport statistics
    if input_data['airport'] in airport_stats.index:
        feature_dict['airport_total_flights'] = airport_stats.loc[input_data['airport'], 'airport_total_flights']
        feature_dict['airport_avg_flights'] = airport_stats.loc[input_data['airport'], 'airport_avg_flights']
        feature_dict['airport_total_cancelled'] = airport_stats.loc[input_data['airport'], 'airport_total_cancelled']
        feature_dict['airport_total_diverted'] = airport_stats.loc[input_data['airport'], 'airport_total_diverted']
    else:
        # Use average values for unknown airports
        feature_dict['airport_total_flights'] = airport_stats['airport_total_flights'].mean()
        feature_dict['airport_avg_flights'] = airport_stats['airport_avg_flights'].mean()
        feature_dict['airport_total_cancelled'] = airport_stats['airport_total_cancelled'].mean()
        feature_dict['airport_total_diverted'] = airport_stats['airport_total_diverted'].mean()
    
    # Carrier statistics
    if input_data['carrier'] in carrier_stats.index:
        feature_dict['carrier_total_flights'] = carrier_stats.loc[input_data['carrier'], 'carrier_total_flights']
        feature_dict['carrier_avg_flights'] = carrier_stats.loc[input_data['carrier'], 'carrier_avg_flights']
        feature_dict['carrier_total_cancelled'] = carrier_stats.loc[input_data['carrier'], 'carrier_total_cancelled']
        feature_dict['carrier_total_diverted'] = carrier_stats.loc[input_data['carrier'], 'carrier_total_diverted']
    else:
        # Use average values for unknown carriers
        feature_dict['carrier_total_flights'] = carrier_stats['carrier_total_flights'].mean()
        feature_dict['carrier_avg_flights'] = carrier_stats['carrier_avg_flights'].mean()
        feature_dict['carrier_total_cancelled'] = carrier_stats['carrier_total_cancelled'].mean()
        feature_dict['carrier_total_diverted'] = carrier_stats['carrier_total_diverted'].mean()
    
    # Create DataFrame
    feature_df = pd.DataFrame([feature_dict])
    
    # Add categorical columns
    feature_df['carrier'] = input_data['carrier']
    feature_df['airport'] = input_data['airport']
    
    # Ensure all required columns exist and are in the right order
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    feature_df = feature_df[feature_columns]
    
    # Encode categorical variables
    categorical_cols = ['carrier', 'airport']
    for col in categorical_cols:
        if col in label_encoders:
            try:
                feature_df[col] = label_encoders[col].transform(feature_df[col].astype(str))
            except ValueError:
                # Handle unseen categories by using the most frequent class
                feature_df[col] = 0
    
    # Scale numerical features
    numerical_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numerical_cols] = scaler.transform(feature_df[numerical_cols])
    
    return feature_df

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Model Prediction"])

# Load data and model
df = load_data()
model, pipeline_data = load_model_and_components()

if df is None:
    st.stop()

# Page 1: Data Exploration
if page == "Data Exploration":
    st.title("✈️ Flight Delay Analysis - Data Exploration")
    st.markdown("---")
    
    # Dataset Overview
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Time Period", f"{df['year'].min()}-{df['year'].max()}")
    with col4:
        st.metric("Airlines", df['carrier'].nunique())
    
    # Data sample
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Missing values analysis
    st.header("🔍 Data Quality Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data.plot(kind='bar', ax=ax)
            plt.title('Missing Values by Column')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.success("No missing values found!")
    
    with col2:
        st.subheader("Data Types")
        dtype_counts = df.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        dtype_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        plt.title('Distribution of Data Types')
        st.pyplot(fig)
    
    # Delay Analysis
    st.header("⏰ Delay Analysis")
    
    # Create delay rate for analysis
    df_analysis = df.copy()
    df_analysis['delay_rate'] = df_analysis['arr_del15'] / df_analysis['arr_flights'].replace(0, 1)
    df_analysis = df_analysis.dropna(subset=['delay_rate'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Arrival Delay Distribution")
        fig = px.histogram(df_analysis, x='arr_delay', nbins=50, 
                          title='Distribution of Arrival Delays')
        fig.update_layout(xaxis_title='Delay Minutes', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Delay Rate Distribution")
        fig = px.histogram(df_analysis, x='delay_rate', nbins=50,
                          title='Distribution of Delay Rates')
        fig.update_layout(xaxis_title='Delay Rate', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trends
    st.header("📅 Temporal Patterns")
    
    monthly_stats = df_analysis.groupby('month').agg({
        'arr_delay': 'mean',
        'delay_rate': 'mean',
        'arr_cancelled': 'mean',
        'arr_diverted': 'mean'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Delay by Month")
        fig = px.bar(x=monthly_stats.index, y=monthly_stats['arr_delay'],
                     title='Average Arrival Delay by Month')
        fig.update_layout(xaxis_title='Month', yaxis_title='Average Delay (minutes)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Delay Rate by Month")
        fig = px.line(x=monthly_stats.index, y=monthly_stats['delay_rate'],
                      title='Delay Rate by Month', markers=True)
        fig.update_layout(xaxis_title='Month', yaxis_title='Delay Rate')
        st.plotly_chart(fig, use_container_width=True)
    
    # Carrier Analysis
    st.header("🏢 Airline Performance")
    
    # Top carriers by volume
    top_carriers = df['carrier'].value_counts().head(10)
    carrier_stats = df_analysis.groupby('carrier').agg({
        'arr_delay': 'mean',
        'delay_rate': 'mean',
        'arr_flights': 'sum'
    }).round(3)
    
    # Filter for top carriers
    top_carrier_stats = carrier_stats.loc[top_carriers.index]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Volume by Carrier (Top 10)")
        fig = px.bar(x=top_carriers.index, y=top_carriers.values,
                     title='Total Flights by Carrier')
        fig.update_layout(xaxis_title='Carrier', yaxis_title='Total Flights')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Delay by Carrier (Top 10)")
        fig = px.bar(x=top_carrier_stats.index, y=top_carrier_stats['arr_delay'],
                     title='Average Delay by Carrier')
        fig.update_layout(xaxis_title='Carrier', yaxis_title='Average Delay (minutes)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Delay causes analysis
    st.header("🔍 Delay Causes Analysis")
    
    delay_causes = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
    delay_means = df[delay_causes].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Delay by Cause")
        fig = px.bar(x=delay_means.index, y=delay_means.values,
                     title='Average Delay Minutes by Cause')
        fig.update_layout(xaxis_title='Delay Cause', yaxis_title='Average Minutes')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Delay Cause Distribution")
        fig = px.pie(values=delay_means.values, names=delay_means.index,
                     title='Distribution of Delay Causes')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.header("🔗 Feature Correlations")
    
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)

# Page 2: Model Prediction
elif page == "Model Prediction":
    st.title("🤖 Flight Delay Prediction")
    st.markdown("---")
    
    if model is None or pipeline_data is None:
        st.error("Model not loaded. Please check if model files exist.")
        st.stop()
    
    # Display model information
    st.header("📋 Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", pipeline_data['model_name'])
    with col2:
        st.metric("F1-Score", f"{pipeline_data['model_performance']['f1_score']:.4f}")
    with col3:
        st.metric("Accuracy", f"{pipeline_data['model_performance']['accuracy']:.4f}")
    
    st.header("📝 Input Flight Information")
    st.markdown("Enter the flight details below to predict if there will be high delay rates (>25%):")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Information")
        year_options = sorted(df['year'].unique())
        year = st.selectbox("Year", options=year_options, 
                        index=len(year_options)-1)  # Select the last (most recent) year
        month = st.selectbox("Month", options=list(range(1, 13)), index=5)  # Index 5 = June (month 6)
        carrier = st.selectbox("Airline Carrier", options=sorted(df['carrier'].unique()), index=0)
        airport = st.selectbox("Airport Code", options=sorted(df['airport'].unique()), index=0)
    
    with col2:
        st.subheader("Flight Operations")
        arr_flights = st.number_input("Number of Arriving Flights", 
                                     min_value=1, max_value=10000, value=100)
        arr_cancelled = st.number_input("Cancelled Flights", 
                                       min_value=0, max_value=int(arr_flights), value=0)
        arr_diverted = st.number_input("Diverted Flights", 
                                      min_value=0, max_value=int(arr_flights), value=0)
    
    with col3:
        st.subheader("Calculated Features")
        # Calculate derived features
        quarter = ((month - 1) // 3) + 1
        is_winter = 1 if month in [12, 1, 2] else 0
        is_summer = 1 if month in [6, 7, 8] else 0
        is_peak_travel = 1 if month in [6, 7, 8, 11, 12] else 0
        flights_per_day = arr_flights / 30
        cancellation_rate = arr_cancelled / arr_flights if arr_flights > 0 else 0
        total_disruptions = arr_cancelled + arr_diverted
        
        st.write(f"**Quarter:** {quarter}")
        st.write(f"**Winter Season:** {'Yes' if is_winter else 'No'}")
        st.write(f"**Summer Season:** {'Yes' if is_summer else 'No'}")
        st.write(f"**Peak Travel Period:** {'Yes' if is_peak_travel else 'No'}")
        st.write(f"**Flights per Day:** {flights_per_day:.1f}")
        st.write(f"**Cancellation Rate:** {cancellation_rate:.3f}")
        st.write(f"**Total Disruptions:** {total_disruptions}")
    
    # Prepare input data
    input_data = {
        'year': year,
        'month': month,
        'carrier': carrier,
        'airport': airport,
        'arr_flights': arr_flights,
        'arr_cancelled': arr_cancelled,
        'arr_diverted': arr_diverted
    }
    
    # Prediction button
    if st.button("🔮 Predict Delay Risk", type="primary"):
        try:
            # Prepare data for prediction
            processed_data = prepare_input_for_prediction(input_data, pipeline_data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            # Display results
            st.header("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **HIGH DELAY RISK PREDICTED**")
                    st.write("The model predicts that this month will have high delay rates (>25%)")
                else:
                    st.success("✅ **NORMAL OPERATIONS PREDICTED**")
                    st.write("The model predicts normal delay rates for this month")
            
            with col2:
                st.subheader("Prediction Confidence")
                prob_no_delay = prediction_proba[0] * 100
                prob_high_delay = prediction_proba[1] * 100
                
                st.write(f"**Normal Operations:** {prob_no_delay:.1f}%")
                st.write(f"**High Delay Risk:** {prob_high_delay:.1f}%")
                
                # Confidence bar
                fig = go.Figure(data=[
                    go.Bar(name='Normal', x=['Prediction'], y=[prob_no_delay], marker_color='green'),
                    go.Bar(name='High Delay', x=['Prediction'], y=[prob_high_delay], marker_color='red')
                ])
                fig.update_layout(barmode='stack', title='Prediction Confidence',
                                yaxis_title='Probability (%)', showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Historical context
            st.header("📈 Historical Context")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Carrier {carrier} Historical Performance")
                carrier_history = df[df['carrier'] == carrier].groupby('month').agg({
                    'arr_delay': 'mean',
                    'arr_del15': 'sum',
                    'arr_flights': 'sum'
                })
                if len(carrier_history) > 0:
                    carrier_history['delay_rate'] = carrier_history['arr_del15'] / carrier_history['arr_flights']
                    
                    fig = px.line(x=carrier_history.index, y=carrier_history['delay_rate'],
                                 title=f'Historical Delay Rate - {carrier}', markers=True)
                    fig.update_layout(xaxis_title='Month', yaxis_title='Delay Rate')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No historical data available for this carrier.")
            
            with col2:
                st.subheader(f"Airport {airport} Historical Performance")
                airport_history = df[df['airport'] == airport].groupby('month').agg({
                    'arr_delay': 'mean',
                    'arr_del15': 'sum',
                    'arr_flights': 'sum'
                })
                if len(airport_history) > 0:
                    airport_history['delay_rate'] = airport_history['arr_del15'] / airport_history['arr_flights']
                    
                    fig = px.line(x=airport_history.index, y=airport_history['delay_rate'],
                                 title=f'Historical Delay Rate - {airport}', markers=True)
                    fig.update_layout(xaxis_title='Month', yaxis_title='Delay Rate')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No historical data available for this airport.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check your input values and try again.")
            st.write("Debug info:", str(e))