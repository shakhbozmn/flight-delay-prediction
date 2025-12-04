import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any

class ChartRenderer:
    
    def __init__(self, theme: str = "plotly"):
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3
    
    def render_metric_cards(self, metrics: Dict[str, Any]):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Records",
                value=f"{metrics['total_records']:,}",
                help="Total number of flight records in the dataset"
            )
        
        with col2:
            st.metric(
                label="Airlines",
                value=f"{metrics['total_airlines']:,}",
                help="Number of unique airline carriers"
            )
        
        with col3:
            st.metric(
                label="Airports",
                value=f"{metrics['total_airports']:,}",
                help="Number of unique airports"
            )
        
        with col4:
            st.metric(
                label="Time Period",
                value=metrics['time_period'],
                help="Data collection period"
            )
    
    def render_delay_distribution(self, df: pd.DataFrame):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, x='arr_delay', nbins=50,
                title='Arrival Delay Distribution',
                template=self.theme,
                color_discrete_sequence=self.color_palette
            )
            fig.update_layout(
                xaxis_title='Delay Minutes',
                yaxis_title='Count',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, x='delay_rate', nbins=50,
                title='Delay Rate Distribution',
                template=self.theme,
                color_discrete_sequence=self.color_palette
            )
            fig.update_layout(
                xaxis_title='Delay Rate',
                yaxis_title='Count',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_monthly_trends(self, monthly_stats: pd.DataFrame):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=monthly_stats.index,
                y=monthly_stats['arr_delay'],
                title='Average Delay by Month',
                template=self.theme,
                color=monthly_stats['arr_delay'],
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title='Average Delay (minutes)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                x=monthly_stats.index,
                y=monthly_stats['delay_rate'],
                title='Delay Rate Trend by Month',
                markers=True,
                template=self.theme
            )
            fig.update_traces(line_color='#FF6B6B', marker_color='#FF6B6B')
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title='Delay Rate',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_carrier_analysis(self, carrier_stats: pd.DataFrame, top_carriers: pd.Series):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=top_carriers.index,
                y=top_carriers.values,
                title='Flight Volume by Carrier (Top 10)',
                template=self.theme,
                color=top_carriers.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                xaxis_title='Carrier',
                yaxis_title='Total Flights',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=carrier_stats.index,
                y=carrier_stats['arr_delay'],
                title='Average Delay by Carrier (Top 10)',
                template=self.theme,
                color=carrier_stats['arr_delay'],
                color_continuous_scale='Oranges'
            )
            fig.update_layout(
                xaxis_title='Carrier',
                yaxis_title='Average Delay (minutes)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_delay_causes(self, delay_means: pd.Series):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=delay_means.index,
                y=delay_means.values,
                title='Average Delay by Cause',
                template=self.theme,
                color=delay_means.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title='Delay Cause',
                yaxis_title='Average Minutes',
                showlegend=False
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=delay_means.values,
                names=delay_means.index,
                title='Delay Cause Distribution',
                template=self.theme,
                color_discrete_sequence=self.color_palette
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_heatmap(self, df: pd.DataFrame):
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    def render_prediction_result(self, prediction: int, prediction_proba: np.ndarray):
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("**HIGH DELAY RISK PREDICTED**")
                st.markdown("The model predicts that this month will have **high delay rates (>25%)**")
            else:
                st.success("**NORMAL OPERATIONS PREDICTED**")
                st.markdown("The model predicts **normal delay rates** for this month")
        
        with col2:
            prob_no_delay = prediction_proba[0] * 100
            prob_high_delay = prediction_proba[1] * 100
            
            st.subheader("Prediction Confidence")
            st.write(f"**Normal Operations**: {prob_no_delay:.1f}%")
            st.write(f"**High Delay Risk**: {prob_high_delay:.1f}%")
            
            fig = go.Figure(data=[
                go.Bar(name='Normal Operations', x=['Prediction'], y=[prob_no_delay], marker_color='green'),
                go.Bar(name='High Delay Risk', x=['Prediction'], y=[prob_high_delay], marker_color='red')
            ])
            fig.update_layout(
                barmode='stack', 
                title='Prediction Confidence',
                yaxis_title='Probability (%)', 
                showlegend=True,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_historical_context(self, df: pd.DataFrame, carrier: str, airport: str):
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
                
                fig = px.line(
                    x=carrier_history.index,
                    y=carrier_history['delay_rate'],
                    title=f'Historical Delay Rate - {carrier}',
                    markers=True,
                    template=self.theme
                )
                fig.update_traces(line_color='#4ECDC4', marker_color='#4ECDC4')
                fig.update_layout(xaxis_title='Month', yaxis_title='Delay Rate')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available for this carrier.")
        
        with col2:
            st.subheader(f"Airport {airport} Historical Performance")
            airport_history = df[df['airport'] == airport].groupby('month').agg({
                'arr_delay': 'mean',
                'arr_del15': 'sum',
                'arr_flights': 'sum'
            })
            
            if len(airport_history) > 0:
                airport_history['delay_rate'] = airport_history['arr_del15'] / airport_history['arr_flights']
                
                fig = px.line(
                    x=airport_history.index,
                    y=airport_history['delay_rate'],
                    title=f'Historical Delay Rate - {airport}',
                    markers=True,
                    template=self.theme
                )
                fig.update_traces(line_color='#FF6B6B', marker_color='#FF6B6B')
                fig.update_layout(xaxis_title='Month', yaxis_title='Delay Rate')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available for this airport.")
    
    def render_model_performance_card(self, performance_data: Dict[str, float], model_name: str):
        st.markdown("### Model Performance Dashboard")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_config = [
            ("Accuracy", performance_data['accuracy'], "Overall correctness"),
            ("F1-Score", performance_data['f1_score'], "Balance of precision & recall"),
            ("Precision", performance_data['precision'], "Accuracy of delay predictions"),
            ("Recall", performance_data['recall'], "Ability to catch actual delays"),
            ("ROC-AUC", performance_data['roc_auc'], "Overall ranking ability")
        ]
        
        cols = [col1, col2, col3, col4, col5]
        
        for i, (metric_name, value, icon, description) in enumerate(metrics_config):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{icon} {metric_name}</h3>
                    <h2>{value:.3f}</h2>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance interpretation
        st.markdown("### Performance Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Model**: {model_name}
            
            **Strengths**:
            - {'High accuracy' if performance_data['accuracy'] > 0.8 else 'Good accuracy'} ({performance_data['accuracy']:.1%})
            - {'Excellent' if performance_data['f1_score'] > 0.7 else 'Good'} F1-Score ({performance_data['f1_score']:.3f})
            - {'Strong' if performance_data['roc_auc'] > 0.8 else 'Good'} ranking ability (AUC: {performance_data['roc_auc']:.3f})
            """)
        
        with col2:
            st.success(f"""
            **Business Impact**:
            - Catches {performance_data['recall']:.1%} of actual high-delay periods
            - {performance_data['precision']:.1%} of delay predictions are correct
            - Reliable for operational planning decisions
            """)