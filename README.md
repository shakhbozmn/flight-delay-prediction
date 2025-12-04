# Flight Delay Prediction and Analysis

## Overview

This project represents ML analysis of airline delays using the U.S. Department of Transportation's Airline On-Time Statistics and Delay Causes dataset. The project core idea is to predict high-delay periods (months with >25% delay rate) and provides insights for airline operations.



## Dataset

The dataset contains **409,612 flight records** from 2003 to 2025, covering domestic flights operated by major U.S. carriers. The dataset includes:

- **Flight Operations**: Total arrivals, cancellations, diversions
- **Delay Causes**: Air Carrier, Weather, National Aviation System, Late Aircraft, Security delays
- **Temporal Information**: Year and month
- **Location Data**: Carrier codes, carrier names, airport codes, airport names
- **Delay Metrics**: Count and duration of delays by cause

## Project Objectives

1. **Predictive Modeling**: Classify months with high delay rates (>25%) vs. normal operations
2. **Data Exploration**: Identify patterns and factors contributing to flight delays
3. **Model Comparison**: Evaluate multiple machine learning algorithms
4. **Feature Engineering**: Create meaningful predictors from raw data
5. **Deployment**: Interactive Streamlit application for exploration and prediction

## Installation

### Prerequisites

- Python 3.11 or higher #3.11.11 version were used while developing
- pip package manager

### Setup Steps

1. **Clone or download the project**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure dataset is in place**
   - The dataset file `Airline_Delay_Cause.csv` should be in the `data/` directory

## Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   - Navigate to `flight_delay_analysis.ipynb`
   - Run cells sequentially to reproduce the analysis

3. **Notebook Sections**:
   - **Data Loading**: Load and inspect the dataset
   - **Exploratory Data Analysis**: Statistical summaries, visualizations, correlation analysis
   - **Data Preparation**: Missing value handling, scaling, feature engineering, train/test split
   - **Model Training**: Training of 5 models (Logistic Regression, Random Forest, Decision Tree, KNN, XGBoost)
   - **Hyperparameter Tuning**: GridSearchCV for Random Forest and XGBoost (best models)
   - **Model Evaluation**: Performance metrics and visualizations

### Running the Streamlit Application

1. **Start the Streamlit app**
   ```bash
   streamlit run src/app.py
   ```

2. **Access the application**
   - The app will open in your default web browser
   - Default URL: `http://localhost:8501`

3. **Application Pages**:
   - **Overview**: Project summary and best model performance
   - **Data Exploration**: Interactive visualizations of delay patterns, carrier performance, and correlations
   - **Model Prediction**: Input form to predict delay risk for specific month-carrier-airport combinations

## Project Sections

### A. Data Loading
- Single dataset source: `Airline_Delay_Cause.csv`
- Dataset shape: 409,612 rows × 21 columns
- Initial data inspection and basic statistics

### B. Exploratory Data Analysis
- **Statistical Summaries**: 
  - Numerical columns: mean, std, min, max, quartiles
  - Categorical columns: unique values, frequency counts
- **Correlation Matrix**: Heatmap showing relationships between numerical features
- **Visualizations**:
  - Histogram: Arrival delay distribution
  - Box plots: Delay distribution by carrier
  - Bar charts: Average delay by month, delay causes, cancellation/diversion rates

### C. Data Preparation
- **Missing Values**: 
  - Delay columns filled with 0 (no delays)
  - Count columns filled with 0
  - Categorical columns filled with 'Unknown'
- **Error Correction**:
  - Negative flight counts corrected
  - Impossible values (delayed flights > total flights) fixed
  - Outlier handling using IQR method
- **Feature Engineering**:
  - Target variable: `high_delay_month` (1 if delay rate > 25%, 0 otherwise)
  - Time-based features: quarter, is_winter, is_summer, is_peak_travel
  - Operational features: flights_per_day, cancellation_rate, total_disruptions
  - Aggregated features: airport and carrier statistics (total flights, cancellations, diversions)
- **Data Splitting**: 
  - Train set: 80% (327,689 samples)
  - Test set: 20% (81,923 samples)
  - Stratified split to maintain class distribution

### D. Model Training
- **Baseline Models**:
  1. Logistic Regression
  2. Random Forest
  3. Decision Tree
  4. K-Nearest Neighbors
  5. XGBoost
- **Hyperparameter Tuning**:
  - Random Forest: GridSearchCV with parameters for n_estimators, max_depth, min_samples_split
  - XGBoost: GridSearchCV with parameters for n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Training Method**: Cross-validation used for model selection

### E. Model Evaluation
- **Evaluation Metrics**:
  - Accuracy
  - F1-Score
  - Precision
  - Recall
  - ROC-AUC
- **Test Set Results**:
  - All models evaluated on held-out test set
  - Best model: XGBoost (F1: 0.667, Accuracy: 79.3%, ROC-AUC: 0.873, Precision: 0.592, Recall: 0.765)
- **Visualizations**:
  - Model comparison charts (Accuracy, F1-Score, Recall, Precision, ROC-AUC, Training Time)
  - ROC curves for all models
  - Confusion matrix for best model
  - Classification report

### F. Deployment
- **Streamlit Application**: Multi-page web application
  - **Overview Page**: Project description and best model metrics
  - **Data Exploration Page**: 
    - Dataset overview metrics
    - Delay distribution charts
    - Monthly trends
    - Carrier performance analysis
    - Delay causes visualization
    - Correlation heatmap
  - **Model Prediction Page**:
    - Input form for year, month, carrier, airport, and operational forecasts
    - Real-time prediction with confidence scores
    - Historical context visualization

## Notes

- The trained models are saved in the `models/` directory
- The notebook should be run from the project root directory to ensure correct file paths
- The Streamlit app requires the dataset and model files to be in their respective directories
- All preprocessing steps are saved in `pipeline_data.pkl` for consistent predictions in the Streamlit app

---

*This project is maintained as a personal machine-learning portfolio project.*
