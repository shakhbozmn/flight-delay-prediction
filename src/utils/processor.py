import pandas as pd
import numpy as np
from typing import Dict, Any

class DataProcessor:
    def __init__(self, pipeline_data: Dict[str, Any]):
        self.pipeline_data = pipeline_data
        self.label_encoders = pipeline_data['label_encoders']
        self.scaler = pipeline_data['scaler']
        self.feature_columns = pipeline_data['feature_columns']
        self.airport_stats = pipeline_data['airport_stats']
        self.carrier_stats = pipeline_data['carrier_stats']
    
    def create_analysis_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_analysis = df.copy()
        df_analysis['delay_rate'] = df_analysis['arr_del15'] / df_analysis['arr_flights'].replace(0, 1)
        return df_analysis.dropna(subset=['delay_rate'])
    
    def get_monthly_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby('month').agg({
            'arr_delay': 'mean',
            'delay_rate': 'mean',
            'arr_cancelled': 'mean',
            'arr_diverted': 'mean'
        }).round(2)
    
    def get_carrier_statistics(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        top_carriers = df['carrier'].value_counts().head(top_n)
        carrier_stats = df.groupby('carrier').agg({
            'arr_delay': 'mean',
            'delay_rate': 'mean',
            'arr_flights': 'sum'
        }).round(3)
        return carrier_stats.loc[top_carriers.index], top_carriers
    
    def get_delay_causes_statistics(self, df: pd.DataFrame) -> pd.Series:
        delay_causes = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
        return df[delay_causes].mean()
    
    def prepare_prediction_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        # Create feature dictionary
        feature_dict = self._create_feature_dict(input_data)
        
        feature_df = pd.DataFrame([feature_dict])
        
        feature_df['carrier'] = input_data['carrier']
        feature_df['airport'] = input_data['airport']
        
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[self.feature_columns]
        
        feature_df = self._encode_categorical_features(feature_df)
        
        feature_df = self._scale_features(feature_df)
        
        return feature_df
    
    def _create_feature_dict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        feature_dict = {
            'year': input_data['year'],
            'month': input_data['month'],
            'arr_flights': input_data['arr_flights'],
            'arr_cancelled': input_data['arr_cancelled'],
            'arr_diverted': input_data['arr_diverted'],
            'quarter': ((input_data['month'] - 1) // 3) + 1,
            'is_winter': 1 if input_data['month'] in [12, 1, 2] else 0,
            'is_summer': 1 if input_data['month'] in [6, 7, 8] else 0,
            'is_peak_travel': 1 if input_data['month'] in [6, 7, 8, 11, 12] else 0,
            'flights_per_day': input_data['arr_flights'] / 30,
            'cancellation_rate': input_data['arr_cancelled'] / input_data['arr_flights'] if input_data['arr_flights'] > 0 else 0,
            'total_disruptions': input_data['arr_cancelled'] + input_data['arr_diverted']
        }
        
        if input_data['airport'] in self.airport_stats.index:
            airport_data = self.airport_stats.loc[input_data['airport']]
            feature_dict.update({
                'airport_total_flights': airport_data['airport_total_flights'],
                'airport_avg_flights': airport_data['airport_avg_flights'],
                'airport_total_cancelled': airport_data['airport_total_cancelled'],
                'airport_total_diverted': airport_data['airport_total_diverted']
            })
        else:
            feature_dict.update({
                'airport_total_flights': self.airport_stats['airport_total_flights'].mean(),
                'airport_avg_flights': self.airport_stats['airport_avg_flights'].mean(),
                'airport_total_cancelled': self.airport_stats['airport_total_cancelled'].mean(),
                'airport_total_diverted': self.airport_stats['airport_total_diverted'].mean()
            })
        
        if input_data['carrier'] in self.carrier_stats.index:
            carrier_data = self.carrier_stats.loc[input_data['carrier']]
            feature_dict.update({
                'carrier_total_flights': carrier_data['carrier_total_flights'],
                'carrier_avg_flights': carrier_data['carrier_avg_flights'],
                'carrier_total_cancelled': carrier_data['carrier_total_cancelled'],
                'carrier_total_diverted': carrier_data['carrier_total_diverted']
            })
        else:
            feature_dict.update({
                'carrier_total_flights': self.carrier_stats['carrier_total_flights'].mean(),
                'carrier_avg_flights': self.carrier_stats['carrier_avg_flights'].mean(),
                'carrier_total_cancelled': self.carrier_stats['carrier_total_cancelled'].mean(),
                'carrier_total_diverted': self.carrier_stats['carrier_total_diverted'].mean()
            })
        
        return feature_dict
    
    def _encode_categorical_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = ['carrier', 'airport']
        for col in categorical_cols:
            if col in self.label_encoders:
                try:
                    feature_df[col] = self.label_encoders[col].transform(feature_df[col].astype(str))
                except ValueError:
                    feature_df[col] = 0
        return feature_df
    
    def _scale_features(self, feature_df: pd.DataFrame) -> np.ndarray:
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numerical_cols] = self.scaler.transform(feature_df[numerical_cols])
        return feature_df