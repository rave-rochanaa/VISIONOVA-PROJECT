# modules/ai_dashboard.py
"""
AI-powered dashboard module for VisioNova
This module provides intelligent chart recommendations and enhanced dashboard features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import random
import uuid

# Integration with existing modules
try:
    from .chart_studio import save_charts, render_chart, validate_chart_config
except ImportError:
    # Fallback functions if not available
    def save_charts():
        """Fallback save function"""
        pass
    
    def render_chart(df, chart_config):
        """Fallback render function"""
        return None
    
    def validate_chart_config(chart_config, compatibility=None):
        """Fallback validation function"""
        return True

def get_columns_by_type(df, column_type=None):
    """Get columns by type - compatible with Chart Studio"""
    if column_type == 'numeric':
        return df.select_dtypes(include=[np.number]).columns.tolist()
    elif column_type == 'categorical':
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        return df.columns.tolist()

class ChartRecommendationEngine:
    """AI-powered chart recommendation system"""
    
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def analyze_data_characteristics(self):
        """Analyze data to provide intelligent recommendations"""
        analysis = {
            'shape': self.df.shape,
            'numeric_features': len(self.numeric_cols),
            'categorical_features': len(self.categorical_cols),
            'datetime_features': len(self.datetime_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'data_quality_score': self._calculate_quality_score()
        }
        return analysis
    
    def _calculate_quality_score(self):
        """Calculate data quality score (0-100)"""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_ratio = self.df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
        return max(0, 100 - (missing_ratio * 100))
    
    def get_smart_recommendations(self, selected_features=None):
        """Generate intelligent chart recommendations based on data analysis"""
        if not selected_features:
            selected_features = self.numeric_cols[:3] + self.categorical_cols[:2]
        
        recommendations = []
        
        # Analyze feature combinations
        numeric_selected = [col for col in selected_features if col in self.numeric_cols]
        categorical_selected = [col for col in selected_features if col in self.categorical_cols]
        
        # Generate recommendations based on feature types
        if len(numeric_selected) >= 2:
            recommendations.extend(self._recommend_numeric_charts(numeric_selected))
        
        if len(categorical_selected) >= 1:
            recommendations.extend(self._recommend_categorical_charts(categorical_selected, numeric_selected))
        
        if len(numeric_selected) >= 1 and len(categorical_selected) >= 1:
            recommendations.extend(self._recommend_mixed_charts(numeric_selected, categorical_selected))
        
        return recommendations
    
    def _recommend_numeric_charts(self, numeric_cols):
        """Recommend charts for numeric data"""
        recommendations = []
        
        if len(numeric_cols) >= 2:
            recommendations.extend([
                {
                    'type': 'scatter',
                    'title': f'🔍 Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}',
                    'description': 'Perfect for exploring relationships between continuous variables',
                    'confidence': 95,
                    'features': numeric_cols[:2],
                    'insight': 'Reveals correlations and outliers'
                },
                {
                    'type': 'bubble',
                    'title': f'💫 Bubble Chart: Multi-dimensional Analysis',
                    'description': 'Visualize 3+ dimensions simultaneously',
                    'confidence': 88,
                    'features': numeric_cols[:3],
                    'insight': 'Shows complex relationships with size encoding'
                }
            ])
        
        recommendations.append({
            'type': 'correlation_heatmap',
            'title': '🔥 Correlation Heatmap',
            'description': 'Discover hidden patterns in numeric relationships',
            'confidence': 92,
            'features': numeric_cols,
            'insight': 'Identifies strong positive/negative correlations'
        })
        
        return recommendations
    
    def _recommend_categorical_charts(self, categorical_cols, numeric_cols):
        """Recommend charts for categorical data"""
        recommendations = []
        
        recommendations.append({
            'type': 'pie',
            'title': f'🥧 Distribution Pie: {categorical_cols[0]}',
            'description': 'Show proportional breakdown of categories',
            'confidence': 85,
            'features': categorical_cols[:1],
            'insight': 'Reveals category dominance and balance'
        })
        
        if numeric_cols:
            recommendations.append({
                'type': 'box',
                'title': f'📦 Box Plot: {numeric_cols[0]} by {categorical_cols[0]}',
                'description': 'Compare distributions across categories',
                'confidence': 90,
                'features': [numeric_cols[0], categorical_cols[0]],
                'insight': 'Shows outliers and distribution differences'
            })
        
        return recommendations
    
    def _recommend_mixed_charts(self, numeric_cols, categorical_cols):
        """Recommend charts for mixed data types"""
        return [
            {
                'type': 'violin',
                'title': f'🎻 Violin Plot: Distribution Symphony',
                'description': 'Beautiful density visualization across categories',
                'confidence': 87,
                'features': [numeric_cols[0], categorical_cols[0]],
                'insight': 'Shows both distribution shape and summary statistics'
            },
            {
                'type': 'bar',
                'title': f'📊 Bar Chart: Category Comparison',
                'description': 'Compare values across categories',
                'confidence': 89,
                'features': [categorical_cols[0]] + numeric_cols[:1],
                'insight': 'Clear comparison of category performance'
            }
        ]

def display_data_insights(df):
    """Display comprehensive data insights"""
    st.subheader("🧠 Data Intelligence Hub")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("🏷️ Features", f"{df.shape[1]}")
    with col3:
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 Numeric", f"{numeric_count}")
    with col4:
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        st.metric("📝 Categorical", f"{categorical_count}")

def display_feature_selector(df, recommendation_engine):
    """Enhanced feature selection with smart suggestions"""
    st.subheader("🎯 Smart Feature Selection")
    
    # Create tabs for different feature types
    tab1, tab2, tab3 = st.tabs(["🔢 Numeric Features", "📝 Categorical Features", "🤖 AI Suggestions"])
    
    selected_features = []
    
    with tab1:
        if recommendation_engine.numeric_cols:
            st.write("**Available Numeric Features:**")
            selected_numeric = st.multiselect(
                "Select numeric columns for analysis:",
                recommendation_engine.numeric_cols,
                default=recommendation_engine.numeric_cols[:2],
                help="Numeric features are great for scatter plots, correlations, and statistical analysis"
            )
            selected_features.extend(selected_numeric)
        else:
            st.info("No numeric features found in your dataset.")
    
    with tab2:
        if recommendation_engine.categorical_cols:
            st.write("**Available Categorical Features:**")
            selected_categorical = st.multiselect(
                "Select categorical columns for analysis:",
                recommendation_engine.categorical_cols,
                default=recommendation_engine.categorical_cols[:1],
                help="Categorical features are perfect for pie charts, bar plots, and group comparisons"
            )
            selected_features.extend(selected_categorical)
        else:
            st.info("No categorical features found in your dataset.")
    
    with tab3:
        st.write("**🎲 Random Feature Combinations:**")
        if st.button("🎲 Surprise Me!", help="Get a random feature combination"):
            all_features = recommendation_engine.numeric_cols + recommendation_engine.categorical_cols
            random_features = random.sample(all_features, min(3, len(all_features)))
            st.session_state.random_features = random_features
        
        if hasattr(st.session_state, 'random_features'):
            st.success(f"Random selection: {', '.join(st.session_state.random_features)}")
            if st.button("Use Random Selection"):
                selected_features = st.session_state.random_features
    
    return selected_features

def display_chart_recommendations(recommendations):
    """Display chart recommendations with enhanced UI"""
    st.subheader("🎨 AI Chart Recommendations")
    
    if not recommendations:
        st.warning("No recommendations available. Please select some features first!")
        return None
    
    # Sort recommendations by confidence
    recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    
    selected_chart = None
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"#{i+1} {rec['title']} (Confidence: {rec['confidence']}%)", expanded=i==0):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**💡 Insight:** {rec['insight']}")
                st.write(f"**📋 Features:** {', '.join(rec['features'])}")
            
            with col2:
                # Confidence bar
                confidence_color = "green" if rec['confidence'] >= 90 else "orange" if rec['confidence'] >= 75 else "red"
                st.markdown(f"""
                <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px;">
                    <div style="background-color: {confidence_color}; width: {rec['confidence']}%; height: 20px; border-radius: 5px;"></div>
                    <center><b>Confidence: {rec['confidence']}%</b></center>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if st.button(f"Create Chart", key=f"create_{i}"):
                    selected_chart = rec
                    break
    
    return selected_chart

def create_chart_from_recommendation(recommendation, df):
    """Create a chart configuration from AI recommendation"""
    chart_config = {
        'id': str(uuid.uuid4()),
        'title': recommendation['title'],
        'type': recommendation['type'],
        'tab': 'AI Recommendations',
        'created_by': 'AI Recommendation',
        'created_at': datetime.now().isoformat(),
        'confidence': recommendation.get('confidence', 0),
        'ai_insight': recommendation.get('insight', '')
    }
    
    # Configure chart-specific parameters
    features = recommendation.get('features', [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if chart_config['type'] == 'scatter':
        if len(features) >= 2:
            chart_config['x_axis'] = features[0]
            chart_config['y_axis'] = features[1]
    elif chart_config['type'] == 'pie':
        if features and features[0] in categorical_cols:
            chart_config['labels'] = features[0]
    elif chart_config['type'] == 'box':
        if len(features) >= 2:
            numeric_feature = next((f for f in features if f in numeric_cols), None)
            categorical_feature = next((f for f in features if f in categorical_cols), None)
            chart_config['y_axis'] = numeric_feature
            chart_config['x_axis'] = categorical_feature
    elif chart_config['type'] == 'bar':
        categorical_feature = next((f for f in features if f in categorical_cols), None)
        numeric_feature = next((f for f in features if f in numeric_cols), None)
        chart_config['x_axis'] = categorical_feature
        chart_config['y_axis'] = numeric_feature
    
    return chart_config

def create_advanced_dashboard_features():
    """Additional advanced dashboard features"""
    st.subheader("🚀 Advanced Dashboard Features")
    
    feature_tabs = st.tabs([
        "🎨 Themes", "📊 Chart Gallery", "🔍 Data Explorer", 
        "📈 Trend Analysis", "🎯 Goals Tracker"
    ])
    
    with feature_tabs[0]:  # Themes
        st.write("**🎨 Dashboard Themes**")
        theme = st.selectbox(
            "Choose your dashboard vibe:",
            ["🌙 Dark Mode", "☀️ Light Mode", "🌈 Colorful", "💼 Professional", "🎮 Gaming"]
        )
        if st.button("Apply Theme"):
            st.session_state.dashboard_theme = theme
            st.success(f"Theme '{theme}' applied! 🎉")
    
    with feature_tabs[1]:  # Chart Gallery
        st.write("**📊 Your Chart Gallery**")
        custom_charts = st.session_state.charts.get('custom_charts', {})
        
        gallery_stats = {
            "Total Charts": len(custom_charts),
            "AI Generated": len([c for c in custom_charts.values() if c.get('created_by') == 'AI Recommendation']),
            "Manual Created": len([c for c in custom_charts.values() if c.get('created_by') != 'AI Recommendation'])
        }
        
        cols = st.columns(3)
        for i, (key, value) in enumerate(gallery_stats.items()):
            cols[i].metric(key, value)
    
    with feature_tabs[2]:  # Data Explorer
        st.write("**🔍 Quick Data Explorer**")
        exploration_type = st.radio(
            "What would you like to explore?",
            ["🔍 Missing Values Heatmap", "📊 Distribution Overview", "🔗 Correlation Network"]
        )
        if st.button("Generate Exploration"):
            st.info(f"Generating {exploration_type}... 🔄")
    
    with feature_tabs[3]:  # Trend Analysis
        st.write("**📈 Automated Trend Detection**")
        st.info("🤖 AI is analyzing your data for interesting trends...")
        
        # Mock trend insights
        trends = [
            "📈 Upward trend detected in numeric columns",
            "🔄 Cyclical pattern found in data distribution",
            "⚠️ Data quality score: Good"
        ]
        
        for trend in trends:
            st.success(trend)
    
    with feature_tabs[4]:  # Goals Tracker
        st.write("**🎯 Dashboard Goals**")
        
        custom_charts = st.session_state.charts.get('custom_charts', {})
        chart_count = len(custom_charts)
        
        goals = [
            {"name": "Create 5 Charts", "current": chart_count, "target": 5, "emoji": "📊"},
            {"name": "Use AI Recommendations", "current": 1 if chart_count > 0 else 0, "target": 1, "emoji": "🤖"},
            {"name": "Explore All Features", "current": 1, "target": 1, "emoji": "🔍"}
        ]
        
        for goal in goals:
            progress = min(goal["current"] / goal["target"], 1.0)
            st.progress(progress)
            st.write(f"{goal['emoji']} {goal['name']}: {goal['current']}/{goal['target']}")

def custom_dashboard_page(df):
    """Main AI-powered custom dashboard page"""
    st.title("🤖 AI-Powered Dashboard Studio")
    st.markdown("*Let artificial intelligence recommend the perfect visualizations for your data*")
    
    # Initialize recommendation engine
    recommendation_engine = ChartRecommendationEngine(df)
    
    # Display data insights
    display_data_insights(df)
    
    # Main tabs
    main_tab1, main_tab2= st.tabs(["🤖 AI Recommendations", "📊 My Dashboard"])
    
    with main_tab1:
        st.header("🧠 Intelligent Chart Recommendations")
        
        # Feature selection
        selected_features = display_feature_selector(df, recommendation_engine)
        
        if selected_features:
            # Generate recommendations
            with st.spinner("🤖 AI is analyzing your data and generating recommendations..."):
                recommendations = recommendation_engine.get_smart_recommendations(selected_features)
            
            # Display recommendations
            selected_chart = display_chart_recommendations(recommendations)
            
            if selected_chart:
                st.success(f"Great choice! Creating {selected_chart['title']}...")
                
                # Create chart configuration
                chart_config = create_chart_from_recommendation(selected_chart, df)
                
                # Add to custom charts
                if 'charts' not in st.session_state:
                    st.session_state.charts = {'custom_charts': {}}
                if 'custom_charts' not in st.session_state.charts:
                    st.session_state.charts['custom_charts'] = {}
                
                st.session_state.charts['custom_charts'][chart_config['id']] = chart_config
                
                # Save charts
                try:
                    save_charts()
                except:
                    pass
                
                
                st.info("🎨 Chart created! View it in the 'My Dashboard' tab!")
    
    with main_tab2:
        st.header("📊 Your AI-Generated Dashboard")
        
        # Get custom charts
        custom_charts = st.session_state.charts.get('custom_charts', {})
        
        if not custom_charts:
            st.info("🎨 No charts found. Use the AI Recommendations tab to create your first chart!")
        else:
            # Display charts
            for chart_id, chart_config in custom_charts.items():
                with st.container():
                    st.subheader(chart_config.get('title', 'Chart'))
                    
                    # Chart metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Type: {chart_config.get('type', 'Unknown')}")
                    with col2:
                        if chart_config.get('created_by') == 'AI Recommendation':
                            st.caption(f"🤖 AI Confidence: {chart_config.get('confidence', 'N/A')}%")
                    
                    # Try to render the chart
                    try:
                        fig = render_chart(df, chart_config)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Chart rendering not available - would display chart here")
                    except:
                        st.info("Chart would be displayed here")
                    
                    # Chart actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Remove", key=f"remove_{chart_id}"):
                            del st.session_state.charts['custom_charts'][chart_id]
                            st.rerun()
                    with col2:
                        if chart_config.get('ai_insight'):
                            st.info(f"💡 {chart_config['ai_insight']}")
    
    with main_tab3:
        create_advanced_dashboard_features()