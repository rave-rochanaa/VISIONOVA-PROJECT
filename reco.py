import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from typing import Dict, List, Any

# --- Custom CSS for clean elegant UI (light theme, spacing, typography) ---
CUSTOM_CSS = """
/* General layout */
main .block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-bottom: 3rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    background: #ffffff;
}

/* Typography */
h1, .stTitle, .streamlit-expanderHeader {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: #111827;
}

h1 {
    font-size: 3rem !important;
    margin-bottom: 0.5rem;
}

h2, h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #111827;
}

h2 {
    margin-top: 2.5rem;
    margin-bottom: 1rem;
}

h3 {
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

/* Body text */
div.stText, div.stMarkdown, p, span {
    color: #6b7280;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    line-height: 1.5;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #fafafa;
    padding-top: 1.5rem;
}

/* Cards */
.stCard {
    border-radius: 0.75rem;
    box-shadow: 0 4px 10px rgb(0 0 0 / 0.05);
    background: #ffffff;
    padding: 1rem 1.25rem 1.25rem 1.25rem;
    margin-bottom: 1.5rem;
}

/* Metrics */
.stMetric label {
    color: #374151;
    font-weight: 600;
}

/* Buttons with subtle hover */
.stButton>button {
    background-color: #111827;
    color: #fff;
    border-radius: 0.5rem;
    padding: 0.45rem 1.3rem;
    font-weight: 600;
    transition: background-color 0.3s ease;
    border: none;
}

.stButton>button:hover {
    background-color: #2563eb;
    color: white;
}

/* Divider */
.stDivider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

/* Tabs */
[data-baseweb="tab"] {
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    color: #374151;
}

[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb;
    border-bottom: 3px solid #2563eb;
}

/* Scrollbars */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: #f9fafb;
}
::-webkit-scrollbar-thumb {
  background-color: #cbd5e1;
  border-radius: 3px;
}

/* Inline code styling */
code {
    font-family: 'Fira Mono', monospace;
    background-color: #f3f4f6;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}
"""

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 Smart Dashboard Recommender",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# Initialize session state variables
if 'user_logs' not in st.session_state:
    st.session_state.user_logs = []
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'current_layout' not in st.session_state:
    st.session_state.current_layout = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {'trends': 5, 'comparisons': 5}
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()
if 'viewing_start' not in st.session_state:
    st.session_state.viewing_start = None
if 'selected_dimension' not in st.session_state:
    st.session_state.selected_dimension = None
if 'selected_measure' not in st.session_state:
    st.session_state.selected_measure = None
if 'dimension_values' not in st.session_state:
    st.session_state.dimension_values = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'chart_recommendations' not in st.session_state:
    st.session_state.chart_recommendations = []
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

# Dashboard Layout Configurations
DASHBOARD_LAYOUTS = {
    "ROI Optimization": {
        "description": "Focus on return on investment metrics with trend analysis",
        "charts": ["Line Chart: ROI over Time", "Bar Chart: ROI by Category", "Scatter: ROI vs Cost"],
        "icon": "📈",
        "color": "#10B981",
        "features": ["has_roi", "has_time", "numeric_heavy"]
    },
    "Performance Analytics": {
        "description": "Compare performance across different dimensions", 
        "charts": ["Bar Chart: Performance Comparison", "Pie Chart: Distribution", "Heatmap: Performance Matrix"],
        "icon": "📊",
        "color": "#3B82F6",
        "features": ["categorical_heavy", "has_performance_metrics"]
    },
    "Time Series Analysis": {
        "description": "Track trends and patterns over time",
        "charts": ["Line Chart: Trends", "Area Chart: Cumulative", "Multi-line: Comparisons"],
        "icon": "⏰",
        "color": "#8B5CF6",
        "features": ["has_time", "sequential_data"]
    },
    "Distribution Overview": {
        "description": "Understand data distribution and composition",
        "charts": ["Pie Chart: Categories", "Histogram: Distribution", "Box Plot: Quartiles"],
        "icon": "🥧",
        "color": "#F59E0B",
        "features": ["categorical_heavy", "distribution_focus"]
    }
}

# Chart Recommendation Rules
CHART_RECOMMENDATIONS = {
    "Single Measure": [
        {"type": "Histogram", "description": "Distribution analysis", "icon": "📊"},
        {"type": "Box Plot", "description": "Outlier detection", "icon": "📦"},
        {"type": "Trend Line", "description": "Time-based patterns", "icon": "📈"},
        {"type": "Pie Chart", "description": "Proportion analysis", "icon": "🥧"}
    ],
    "Measure and Dimension": [
        {"type": "Bar Chart", "description": "Category comparison", "icon": "📊"},
        {"type": "Line Chart", "description": "Trends by category", "icon": "📈"},
        {"type": "Scatter Plot", "description": "Relationship analysis", "icon": "🔍"},
        {"type": "Heatmap", "description": "Intensity visualization", "icon": "🔥"}
    ],
    "Multiple Measures": [
        {"type": "Scatter Matrix", "description": "Multi-variable relationships", "icon": "🔢"},
        {"type": "Correlation Heatmap", "description": "Measure relationships", "icon": "📊"},
        {"type": "Parallel Coordinates", "description": "Multi-dimensional analysis", "icon": "📐"},
        {"type": "Bubble Chart", "description": "3-variable comparison", "icon": "🫧"}
    ]
}

# ---------------------------
# 1. User Behavior Tracking System
class BehaviorLogger:
    @staticmethod
    def log_event(event_type: str, details: Dict[str, Any], dataset_metadata: Dict = None):
        """Log user interaction events"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'dataset_metadata': dataset_metadata or {},
            'session_id': str(st.session_state.session_start)
        }
        st.session_state.user_logs.append(event)
        
        # Save to file for persistence
        BehaviorLogger.save_logs()
    
    @staticmethod
    def save_logs():
        """Save logs to JSON file"""
        try:
            with open('user_behavior_logs.json', 'w') as f:
                json.dump(st.session_state.user_logs, f, indent=2)
        except Exception as e:
            st.error("Error saving logs: {e}".format(e=e))

    
    @staticmethod
    def load_logs():
        """Load existing logs"""
        try:
            if os.path.exists('user_behavior_logs.json'):
                with open('user_behavior_logs.json', 'r') as f:
                    st.session_state.user_logs = json.load(f)
        except Exception as e:
            st.error(f"Error loading logs: {e}")

# ---------------------------
# 2. Enhanced Dataset Analysis and Feature Extraction
class DatasetAnalyzer:
    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from uploaded dataset"""
        if df is None or df.empty:
            return {}
            
        # Basic features
        features = {
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'num_numeric': len(df.select_dtypes(include=[np.number]).columns),
            'num_categorical': len(df.select_dtypes(include=['object']).columns),
            'num_datetime': len(df.select_dtypes(include=['datetime64']).columns),
            'has_roi': any('roi' in col.lower() for col in df.columns),
            'has_time': any(keyword in str(df.columns).lower() for keyword in ['date', 'time', 'month', 'year']),
            'has_performance': any(keyword in str(df.columns).lower() for keyword in ['performance', 'score', 'rating']),
            'categorical_ratio': len(df.select_dtypes(include=['object']).columns) / len(df.columns) if len(df.columns) > 0 else 0,
            'numeric_ratio': len(df.select_dtypes(include=[np.number]).columns) / len(df.columns) if len(df.columns) > 0 else 0,
            'dimensions': list(df.select_dtypes(include=['object', 'category']).columns),
            'measures': list(df.select_dtypes(include=[np.number]).columns),
            'datetime_cols': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Additional derived features
        features['numeric_heavy'] = features['numeric_ratio'] > 0.6
        features['categorical_heavy'] = features['categorical_ratio'] > 0.4
        features['small_dataset'] = features['num_rows'] < 100
        features['large_dataset'] = features['num_rows'] > 10000
        features['sequential_data'] = features['has_time'] or features['num_rows'] > 50
        features['distribution_focus'] = features['categorical_heavy'] or features['num_categorical'] > 2
        features['has_performance_metrics'] = features['has_performance'] or features['numeric_heavy']
        
        # Identify key metrics
        features['key_metrics'] = DatasetAnalyzer.identify_key_metrics(df)
        
        return features
    
    @staticmethod
    def identify_key_metrics(df: pd.DataFrame) -> List[str]:
        """Identify the most important metrics in the dataset"""
        if df is None or df.empty:
            return []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Simple heuristic: metrics with highest variance or business relevance
        key_metrics = []
        business_keywords = ['revenue', 'profit', 'sales', 'growth', 'conversion', 'roi', 'kpi']
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Check for business relevance
            if any(kw in col_lower for kw in business_keywords):
                key_metrics.append(col)
                
            # Check for high variance
            elif len(df[col].unique()) > 10 and df[col].std() > df[col].mean() * 0.5:
                key_metrics.append(col)
        
        # Fallback to all numeric columns if no key metrics found
        if not key_metrics and len(numeric_cols) > 0:
            key_metrics = list(numeric_cols[:3])
            
        return key_metrics

# ---------------------------
# 3. Enhanced Machine Learning Model
class RecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def prepare_training_data(self, logs: List[Dict]) -> tuple:
        """Convert logs to training data"""
        X, y = [], []
        
        for log in logs:
            if log['event_type'] in ['layout_applied', 'chart_clicked'] and log.get('dataset_metadata'):
                # Extract features
                metadata = log['dataset_metadata']
                if not metadata:  # Skip if no metadata
                    continue
                    
                feature_vector = [
                    metadata.get('num_rows', 0),
                    metadata.get('num_cols', 0),
                    metadata.get('num_numeric', 0),
                    metadata.get('categorical_ratio', 0),
                    int(metadata.get('has_roi', False)),
                    int(metadata.get('has_time', False)),
                    int(metadata.get('numeric_heavy', False))
                ]
                
                X.append(feature_vector)
                
                # Extract label
                if log['event_type'] == 'layout_applied':
                    y.append(log['details']['layout_name'])
                else:  # chart_clicked
                    # Map chart to most relevant layout
                    chart_name = log['details'].get('chart_name', '').lower()
                    if 'roi' in chart_name:
                        y.append('ROI Optimization')
                    elif 'time' in chart_name or 'trend' in chart_name:
                        y.append('Time Series Analysis')
                    elif 'pie' in chart_name or 'distribution' in chart_name:
                        y.append('Distribution Overview')
                    else:
                        y.append('Performance Analytics')
        
        return np.array(X), np.array(y)
    
    def train(self, logs: List[Dict]) -> bool:
        """Train the recommendation model"""
        if len(logs) < 5:  # Need minimum data
            return False
            
        try:
            X, y = self.prepare_training_data(logs)
            
            if len(X) == 0 or len(set(y)) < 2:
                return False
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train model
            self.model.fit(X, y_encoded)
            self.is_trained = True
            
            # Save model
            with open('recommendation_model.pkl', 'wb') as f:
                pickle.dump({'model': self.model, 'encoder': self.label_encoder}, f)
            
            return True
        except Exception as e:
            st.error(f"Training error: {e}")
            return False
    
    def predict(self, dataset_features: Dict) -> List[tuple]:
        """Make recommendations based on dataset features"""
        if not self.is_trained or not dataset_features:
            return self._rule_based_recommendations(dataset_features)
        
        try:
            # Prepare feature vector
            feature_vector = np.array([[
                dataset_features.get('num_rows', 0),
                dataset_features.get('num_cols', 0),
                dataset_features.get('num_numeric', 0),
                dataset_features.get('categorical_ratio', 0),
                int(dataset_features.get('has_roi', False)),
                int(dataset_features.get('has_time', False)),
                int(dataset_features.get('numeric_heavy', False))
            ]])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(feature_vector)[0]
            layout_names = self.label_encoder.classes_
            
            # Sort by probability
            recommendations = [(layout_names[i], probabilities[i]) 
                             for i in range(len(layout_names))]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return self._rule_based_recommendations(dataset_features)
    
    def _rule_based_recommendations(self, features: Dict) -> List[tuple]:
        """Fallback rule-based recommendations"""
        if not features:
            # Default recommendations when no features available
            return [
                ('Performance Analytics', 0.7),
                ('Distribution Overview', 0.6),
                ('Time Series Analysis', 0.5),
                ('ROI Optimization', 0.4)
            ]
            
        scores = {}
        
        # ROI Optimization scoring
        roi_score = 0.8 if features.get('has_roi') else 0.1
        roi_score += 0.3 if features.get('numeric_heavy') else 0
        roi_score += 0.2 if features.get('has_time') else 0
        scores['ROI Optimization'] = min(roi_score, 1.0)
        
        # Time Series scoring  
        time_score = 0.9 if features.get('has_time') else 0.1
        time_score += 0.2 if features.get('numeric_heavy') else 0
        scores['Time Series Analysis'] = min(time_score, 1.0)
        
        # Performance Analytics scoring
        perf_score = 0.7 if features.get('has_performance') else 0.3
        perf_score += 0.2 if features.get('categorical_heavy') else 0
        scores['Performance Analytics'] = min(perf_score, 1.0)
        
        # Distribution Overview scoring
        dist_score = 0.8 if features.get('categorical_heavy') else 0.2
        scores['Distribution Overview'] = min(dist_score, 1.0)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------------------------
# 4. Analysis Engine
class AnalysisEngine:
    @staticmethod
    def analyze_metric(df: pd.DataFrame, measure: str, dimension: str = None, dimension_values: List[str] = None):
        """Perform comprehensive analysis on a selected metric"""
        if measure not in df.columns:
            return {}
            
        results = {}
        numeric_col = df[measure]
        
        # Basic statistics
        results['basic_stats'] = {
            'mean': numeric_col.mean(),
            'median': numeric_col.median(),
            'min': numeric_col.min(),
            'max': numeric_col.max(),
            'std_dev': numeric_col.std(),
            'variance': numeric_col.var(),
            'skewness': numeric_col.skew(),
            'kurtosis': numeric_col.kurtosis(),
            'q1': numeric_col.quantile(0.25),
            'q3': numeric_col.quantile(0.75)
        }
        
        # Trend analysis if time dimension exists
        time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            try:
                time_series = df.set_index(pd.to_datetime(df[time_col])).resample('M')[measure].mean()
                results['trend_analysis'] = {
                    'trend': 'increasing' if time_series.pct_change().mean() > 0 else 'decreasing',
                    'seasonality': AnalysisEngine.detect_seasonality(time_series)
                }
            except:
                pass
        
        # Dimension-based analysis
        if dimension and dimension in df.columns:
            if dimension_values:
                filtered_df = df[df[dimension].isin(dimension_values)]
            else:
                filtered_df = df
                
            dimension_stats = filtered_df.groupby(dimension)[measure].agg(['mean', 'median', 'std', 'count'])
            results['dimension_analysis'] = dimension_stats.to_dict()
            
            # Identify top performers
            if not dimension_values:
                top_categories = dimension_stats.sort_values('mean', ascending=False).index[:5].tolist()
                results['top_performers'] = top_categories
        
        # Correlation analysis
        corr_matrix = df.corr(numeric_only=True)
        if measure in corr_matrix.columns:
            correlations = corr_matrix[measure].sort_values(key=abs, ascending=False)
            results['correlations'] = correlations.drop(measure).head(5).to_dict()
        
        return results
    
    @staticmethod
    def detect_seasonality(series: pd.Series) -> str:
        """Detect seasonality in time series data"""
        if len(series) < 12:
            return "insufficient data"
            
        # Simple autocorrelation check
        autocorr = [series.autocorr(lag=i) for i in range(1, 13)]
        max_corr = max(autocorr, key=abs)
        
        if abs(max_corr) > 0.5:
            return f"strong seasonality (peak at lag {autocorr.index(max_corr) + 1})"
        elif abs(max_corr) > 0.3:
            return f"moderate seasonality (peak at lag {autocorr.index(max_corr) + 1})"
        return "no significant seasonality detected"
    
    @staticmethod
    def recommend_charts(analysis_context: Dict) -> List[Dict]:
        """Recommend charts based on analysis context"""
        recommendations = []
        measure_count = analysis_context.get('measure_count', 1)
        dimension_count = analysis_context.get('dimension_count', 0)
        time_dimension = analysis_context.get('has_time', False)
        
        # Determine recommendation category
        if measure_count == 1 and dimension_count == 0:
            category = "Single Measure"
        elif measure_count >= 1 and dimension_count >= 1:
            category = "Measure and Dimension"
        elif measure_count > 1:
            category = "Multiple Measures"
        else:
            category = "Single Measure"
        
        # Get base recommendations
        base_recs = CHART_RECOMMENDATIONS.get(category, [])
        
        # Add specialized recommendations based on context
        if time_dimension:
            base_recs.append({"type": "Time Series", "description": "Historical trends", "icon": "⏰"})
        if dimension_count > 1:
            base_recs.append({"type": "Faceted Charts", "description": "Multi-dimensional views", "icon": "🔍"})
        
        # Filter out duplicates
        seen = set()
        for rec in base_recs:
            rec_tuple = (rec['type'], rec['description'])
            if rec_tuple not in seen:
                recommendations.append(rec)
                seen.add(rec_tuple)
        
        return recommendations

# Initialize components
logger = BehaviorLogger()
analyzer = DatasetAnalyzer()
model = RecommendationModel()
analysis_engine = AnalysisEngine()

# Load existing data
logger.load_logs()

# Try to load existing model
try:
    if os.path.exists('recommendation_model.pkl'):
        with open('recommendation_model.pkl', 'rb') as f:
            saved_data = pickle.load(f)
            model.model = saved_data['model']
            model.label_encoder = saved_data['encoder']
            model.is_trained = True
            st.session_state.trained_model = model
except Exception:
    pass

# --- Main Application Interface ---
st.title("🧠 Smart Dashboard Recommender")
st.markdown("### Transform your data into personalized dashboard experiences")

# Sidebar - User Preferences & Stats
with st.sidebar:
    st.header("🎛️ Your Preferences")
    
    # Preference sliders
    trends_pref = st.slider("📈 I prefer trends", 1, 10, st.session_state.user_preferences['trends'])
    comparisons_pref = st.slider("📊 I like comparisons", 1, 10, st.session_state.user_preferences['comparisons'])
    
    if trends_pref != st.session_state.user_preferences['trends'] or comparisons_pref != st.session_state.user_preferences['comparisons']:
        st.session_state.user_preferences = {'trends': trends_pref, 'comparisons': comparisons_pref}
        logger.log_event('preference_updated', {'trends': trends_pref, 'comparisons': comparisons_pref})
    
    st.markdown("---")
    
    # Statistics
    st.header("📊 Your Activity")
    st.metric("Total Interactions", len(st.session_state.user_logs))
    
    if st.session_state.user_logs:
        recent_activities = [log['event_type'] for log in st.session_state.user_logs[-5:]]
        st.write("Recent activities:")
        for activity in recent_activities:
            st.write(f"• {activity.replace('_', ' ').title()}")
    
    st.markdown("---")
    
    # Model Status
    st.header("🤖 AI Status")
    if st.session_state.trained_model and st.session_state.trained_model.is_trained:
        st.success("✅ AI Model Active")
    else:
        st.info("🔄 Learning from your behavior...")
        if len(st.session_state.user_logs) >= 5:
            if st.button("🧠 Train AI Model"):
                if model.train(st.session_state.user_logs):
                    st.session_state.trained_model = model
                    st.success("Model trained successfully!")
                    st.rerun()

# Main content area tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload Data", "🎯 Dashboard Recs", "🔍 Metric Analysis", "📈 Analytics", "🔧 System"])

# Tab 1: Upload Data
with tab1:
    st.header("📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your dataset to get personalized dashboard recommendations"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_dataset = df
            
            # Analyze dataset
            dataset_features = analyzer.analyze_dataset(df)
            
            # Log upload event
            logger.log_event('dataset_uploaded', {
                'filename': uploaded_file.name,
                'size': len(df)
            }, dataset_features)
            
            # Display dataset info in cards
            col1, col2, col3, col4 = st.columns(4, gap="large")
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Numeric Cols", dataset_features['num_numeric'])
            with col4:
                st.metric("Categorical Cols", dataset_features['num_categorical'])
            
            # Show dataset preview
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show detected features blocks
            st.subheader("🔍 Detected Features")
            feature_cols = st.columns(3, gap="medium")
            
            with feature_cols[0]:
                if dataset_features['has_roi']:
                    st.success("💰 ROI columns detected")
                if dataset_features['has_time']:
                    st.success("⏰ Time columns detected")
                    
            with feature_cols[1]:
                if dataset_features['numeric_heavy']:
                    st.info("📊 Numeric-heavy dataset")
                if dataset_features['categorical_heavy']:
                    st.info("📝 Category-rich dataset")
                    
            with feature_cols[2]:
                if dataset_features['has_performance']:
                    st.success("📈 Performance metrics found")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Tab 2: Dashboard Recommendations
with tab2:
    st.header("🎯 Personalized Dashboard Recommendations")
    
    if st.session_state.current_dataset is not None:
        dataset_features = analyzer.analyze_dataset(st.session_state.current_dataset)
        
        # Get recommendations
        if st.session_state.trained_model:
            recommendations = st.session_state.trained_model.predict(dataset_features)
        else:
            recommendations = model.predict(dataset_features)
        
        if recommendations:
            top_rec = recommendations[0]
            layout_name = top_rec[0]
            confidence = top_rec[1]
            
            st.markdown("### 🌟 Top Recommendation")
            layout_info = DASHBOARD_LAYOUTS[layout_name]
            
            col1, col2 = st.columns([2, 1], gap="small")
            
            with col1:
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 12px; background: linear-gradient(135deg, {layout_info['color']}33, {layout_info['color']}1a); border-left: 5px solid {layout_info['color']}; color: #111827; font-weight: 600;'>
                    <h3 style="margin-top: 0;">{layout_info['icon']} {layout_name}</h3>
                    <p style="margin-bottom: 0.25rem;">{layout_info['description']}</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"🚀 Apply {layout_name}", key=f"apply_{layout_name}"):
                    st.session_state.current_layout = layout_name
                    logger.log_event('layout_applied', {'layout_name': layout_name}, dataset_features)
                    st.success(f"✅ Applied {layout_name} layout!")
                    st.session_state.viewing_start = datetime.now()
                
                colfb1, colfb2 = st.columns(2, gap="small")
                with colfb1:
                    if st.button("👍", key=f"like_{layout_name}"):
                        logger.log_event('feedback_positive', {'layout_name': layout_name}, dataset_features)
                        st.success("Thanks for the feedback!")
                with colfb2:
                    if st.button("👎", key=f"dislike_{layout_name}"):
                        logger.log_event('feedback_negative', {'layout_name': layout_name}, dataset_features)
                        st.info("We'll improve our recommendations!")
            
            st.markdown("### 🧠 Why This Recommendation?")
            reasons = []
            if dataset_features.get('has_roi') and 'ROI' in layout_name:
                reasons.append("📊 Your data contains ROI columns")
            if dataset_features.get('has_time') and 'Time Series' in layout_name:
                reasons.append("⏰ Time-based data detected")
            if dataset_features.get('categorical_heavy') and 'Distribution' in layout_name:
                reasons.append("📝 Rich categorical data found")
            if dataset_features.get('numeric_heavy') and 'Performance' in layout_name:
                reasons.append("🔢 Numeric-heavy dataset")
            if not reasons:
                reasons.append("🎯 Based on your interaction patterns and dataset characteristics")
            for reason in reasons:
                st.write(f"• {reason}")
        
        st.markdown("### 📋 Dashboard Layout Recommendations")
        # Display all recommendations in a grid
        cols = st.columns(len(recommendations))
        for i, (layout_name, score) in enumerate(recommendations):
            layout_info = DASHBOARD_LAYOUTS[layout_name]
            with cols[i]:
                with st.container():
                    st.markdown(f"<div style='text-align: center; margin-bottom: 1rem;'>"
                                f"<div style='font-size: 2rem;'>{layout_info['icon']}</div>"
                                f"<h3 style='margin-bottom: 0.5rem;'>{layout_name}</h3>"
                                f"<div style='height: 4px; background: {layout_info['color']}; "
                                f"width: {score*100}%; margin: 0 auto;'></div>"
                                f"<small>Confidence: {score:.0%}</small>"
                                f"</div>", unsafe_allow_html=True)
                    st.caption(layout_info['description'])
                    
                    if st.button("✨ Apply", key=f"apply_{layout_name}_{i}"): 
                               
                        st.session_state.current_layout = layout_name
                        logger.log_event('layout_applied', {'layout_name': layout_name}, dataset_features)
                        st.success(f"✅ Applied {layout_name} layout!")
                        st.session_state.viewing_start = datetime.now()
        
        st.markdown("---")
        st.subheader("📊 Key Metrics Recommendations")
        
        if dataset_features.get('key_metrics'):
            st.info("Based on your dataset, these metrics are likely most important to analyze:")
            
            metric_cols = st.columns(len(dataset_features['key_metrics']))
            for i, metric in enumerate(dataset_features['key_metrics']):
                with metric_cols[i]:
                    st.metric(f"⭐ {metric}", 
                              f"{st.session_state.current_dataset[metric].mean():,.2f}", 
                              delta=f"{st.session_state.current_dataset[metric].std():,.2f} std dev")
        else:
            st.warning("No key metrics identified. Please analyze individual metrics in the Metric Analysis tab.")
        
        st.markdown("---")
        st.subheader("📈 Chart Recommendations")
        
        if dataset_features.get('dimensions') and dataset_features.get('measures'):
            st.success("Recommended charts based on your dataset structure:")
            
            # Prepare analysis context
            analysis_context = {
                'measure_count': len(dataset_features['measures']),
                'dimension_count': len(dataset_features['dimensions']),
                'has_time': dataset_features['has_time']
            }
            
            # Get chart recommendations
            chart_recs = analysis_engine.recommend_charts(analysis_context)
            
            # Display in columns
            rec_cols = st.columns(3)
            for i, rec in enumerate(chart_recs[:6]):  # Show max 6 recommendations
                with rec_cols[i % 3]:
                    with st.container():
                        st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; "
                                    f"border: 1px solid #e5e7eb; margin-bottom: 1rem;'>"
                                    f"<div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>{rec['icon']}</div>"
                                    f"<h4 style='margin-bottom: 0.25rem;'>{rec['type']}</h4>"
                                    f"<p style='font-size: 0.85rem; margin-bottom: 0;'>{rec['description']}</p>"
                                    f"</div>", unsafe_allow_html=True)
        else:
            st.warning("Dataset doesn't have enough dimensions/measures for chart recommendations")
    else:
        st.info("📤 Upload a dataset first to get personalized recommendations!")

# Tab 3: Metric Analysis
with tab3:
    st.header("🔍 Deep Metric Analysis")
    
    if st.session_state.current_dataset is not None:
        df = st.session_state.current_dataset
        dataset_features = analyzer.analyze_dataset(df)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("Select Metric")
            if dataset_features.get('measures'):
                selected_measure = st.selectbox(
                    "Choose a measure to analyze", 
                    options=dataset_features['measures'],
                    index=0,
                    key='selected_measure'
                )
            else:
                st.warning("No numeric measures found in dataset")
                selected_measure = None
        dataset_features = {'dimensions': ['Category', 'Subcategory', 'Region'] } # Example dimensions

                
        with col2:
            st.subheader("Filter by Dimension")
            if dataset_features.get('dimensions'):
                selected_dimension = st.selectbox(
                    "Choose a dimension to filter by", 
                    options=['None'] + dataset_features['dimensions'],
                    index=0,
                    key='selected_dimension'
                )
                
                if selected_dimension and selected_dimension in df.columns:
                    dimension_values = st.multiselect(
                        f"Select {selected_dimension} values",
                        options=df[selected_dimension].unique().tolist(),
                        key='dimension_values'
                    )
                else:
                    st.warning("No categorical dimensions found in the dataset.")
                    selected_dimension = None
                    dimension_values = []
            else:
                st.warning("No categorical dimensions found in dataset")
                selected_dimension = None
                dimension_values = []
            
        
        if selected_measure:
            # Perform analysis
            if 'analysis_results' not in st.session_state or st.session_state.get('last_analysis') != (selected_measure, selected_dimension, tuple(dimension_values)):
                st.session_state.analysis_results = analysis_engine.analyze_metric(
                    df, selected_measure, selected_dimension if selected_dimension != 'None' else None, dimension_values
                )
                st.session_state.last_analysis = (selected_measure, selected_dimension, tuple(dimension_values))
            
            st.markdown("---")
            st.subheader(f"📊 Analysis of {selected_measure}")
            
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                
                # Display basic statistics
                st.markdown("### 📈 Basic Statistics")
                stats_cols = st.columns(4)
                stats = [
                    ("Mean", results['basic_stats']['mean']),
                    ("Median", results['basic_stats']['median']),
                    ("Min", results['basic_stats']['min']),
                    ("Max", results['basic_stats']['max']),
                    ("Std Dev", results['basic_stats']['std_dev']),
                    ("Variance", results['basic_stats']['variance']),
                    ("Skewness", results['basic_stats']['skewness']),
                    ("Kurtosis", results['basic_stats']['kurtosis'])
                ]
                
                for i, (label, value) in enumerate(stats):
                    with stats_cols[i % 4]:
                        st.metric(label, f"{value:,.2f}" if isinstance(value, float) else value)
                
                # Display trend analysis
                if 'trend_analysis' in results:
                    st.markdown("### 📅 Trend Analysis")
                    trend_cols = st.columns(2)
                    with trend_cols[0]:
                        st.metric("Overall Trend", results['trend_analysis']['trend'])
                    with trend_cols[1]:
                        st.metric("Seasonality", results['trend_analysis']['seasonality'])
                
                # Display dimension analysis
                if 'dimension_analysis' in results:
                    st.markdown("### 🔠 Dimension Analysis")
                    dim_df = pd.DataFrame(results['dimension_analysis']).T.reset_index()
                    dim_df = dim_df.rename(columns={'index': selected_dimension})
                    st.dataframe(dim_df.style.format("{:,.2f}"), use_container_width=True)
                    
                    # Show top performers
                    if 'top_performers' in results:
                        st.markdown(f"#### 🏆 Top Performing {selected_dimension} Categories")
                        for i, category in enumerate(results['top_performers'][:5]):
                            st.write(f"{i+1}. {category}")
                
                # Display correlations
                if 'correlations' in results and results['correlations']:
                    st.markdown("### 🔗 Correlation Analysis")
                    corr_df = pd.DataFrame(list(results['correlations'].items()), 
                                          columns=['Metric', 'Correlation'])
                    st.dataframe(corr_df, use_container_width=True)
                
                # Chart recommendations for this metric
                has_time = dataset_features.get('has_time', False) 
                st.markdown("### 📊 Recommended Visualizations")
                analysis_context = {
                    'measure_count': 1,
                    'dimension_count': 1 if selected_dimension and selected_dimension != 'None' else 0,
                    'has_time': has_time
                }
                chart_recs = analysis_engine.recommend_charts(analysis_context)
                
                rec_cols = st.columns(3)
                for i, rec in enumerate(chart_recs[:3]):
                    with rec_cols[i]:
                        with st.container():
                            st.markdown(f"<div style='padding: 1rem; border-radius: 0.5rem; "
                                        f"border: 1px solid #e5e7eb; margin-bottom: 1rem;'>"
                                        f"<div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>{rec['icon']}</div>"
                                        f"<h4 style='margin-bottom: 0.25rem;'>{rec['type']}</h4>"
                                        f"<p style='font-size: 0.85rem; margin-bottom: 0;'>{rec['description']}</p>"
                                        f"</div>", unsafe_allow_html=True)
            else:
                st.warning("No analysis results available for this metric")
        else:
            st.info("Select a metric to analyze")
    else:
        st.info("📤 Upload a dataset first to analyze metrics!")

# Tab 4: Analytics Dashboard
with tab4:
    st.header("📈 Your Dashboard Analytics")
    
    if st.session_state.current_dataset is not None and st.session_state.current_layout:
        layout_info = DASHBOARD_LAYOUTS[st.session_state.current_layout]
        st.markdown(f"### {layout_info['icon']} Current Layout: {st.session_state.current_layout}")
        df = st.session_state.current_dataset
        
        if "ROI" in st.session_state.current_layout:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("📈 Trend Analysis")
                if len(df.select_dtypes(include=[np.number]).columns) >= 2:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
                    fig = px.line(df.head(50), y=numeric_cols[0], title=f"{numeric_cols[0]} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button("🔍 Analyze Trend", key="analyze_trend"):
                        logger.log_event('chart_clicked', {'chart_name': 'Line Chart: Trends'}, analyzer.analyze_dataset(df))
                else:
                    st.info("Not enough numeric columns for trend analysis")
            with col2:
                st.subheader("📊 Performance Comparison")
                if len(df.select_dtypes(include=[np.number]).columns) >= 1:
                    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                    fig = px.histogram(df, x=numeric_col, title=f"Distribution of {numeric_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button("📊 View Distribution", key="view_dist"):
                        logger.log_event('chart_clicked', {'chart_name': 'Histogram: Distribution'}, analyzer.analyze_dataset(df))
                else:
                    st.info("No numeric columns available for distribution analysis")
        
        elif "Time Series" in st.session_state.current_layout:
            st.subheader("⏰ Time Series Dashboard")
            if len(df.select_dtypes(include=[np.number]).columns) >= 1:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:2]:
                    fig = px.line(df.head(100), y=col, title=f"{col} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"🔍 Analyze {col}", key=f"analyze_{col}"):
                        logger.log_event('chart_clicked', {'chart_name': f'Line Chart: {col}'}, analyzer.analyze_dataset(df))
            else:
                st.info("No numeric columns available for time series analysis")
        
        elif "Distribution" in st.session_state.current_layout:
            st.subheader("🥧 Distribution Analysis")
            col1, col2 = st.columns(2, gap="large")
            with col1:
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    cat_col = cat_cols[0]
                    value_counts = df[cat_col].value_counts().head(10)
                    fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                 title=f"Distribution of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"🥧 Explore {cat_col}", key=f"explore_{cat_col}"):
                        logger.log_event('chart_clicked', {'chart_name': f'Pie Chart: {cat_col}'}, analyzer.analyze_dataset(df))
                else:
                    st.info("No categorical columns available for pie chart")
            with col2:
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    num_col = num_cols[0]
                    fig = px.box(df, y=num_col, title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"📦 Analyze {num_col}", key=f"box_{num_col}"):
                        logger.log_event('chart_clicked', {'chart_name': f'Box Plot: {num_col}'}, analyzer.analyze_dataset(df))
                else:
                    st.info("No numeric columns available for box plot")
        
        else:  # Performance Analytics
            st.subheader("📊 Performance Analytics")
            col1, col2 = st.columns(2, gap="large")
            with col1:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 1:
                    numeric_col = numeric_cols[0]
                    fig = px.bar(df.head(20), y=numeric_col, title=f"Top 20 {numeric_col} Values")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"📊 Analyze {numeric_col}", key=f"bar_{numeric_col}"):
                        logger.log_event('chart_clicked', {'chart_name': f'Bar Chart: {numeric_col}'}, analyzer.analyze_dataset(df))
                else:
                    st.info("No numeric columns available for bar chart")
            with col2:
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) >= 1:
                    cat_col = cat_cols[0]
                    value_counts = df[cat_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Count by {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"📈 Explore {cat_col}", key=f"count_{cat_col}"):
                        logger.log_event('chart_clicked', {'chart_name': f'Count Chart: {cat_col}'}, analyzer.analyze_dataset(df))
                else:
                    st.info("No categorical columns available for count chart")
        
        # Track viewing time and log extended viewing events
        if st.session_state.viewing_start:
            viewing_duration = (datetime.now() - st.session_state.viewing_start).total_seconds()
            if viewing_duration > 10:  # Log after 10 secs
                logger.log_event('extended_viewing', {
                    'layout_name': st.session_state.current_layout,
                    'duration_seconds': viewing_duration
                }, analyzer.analyze_dataset(df))
                st.session_state.viewing_start = None
    else:
        st.info("📊 Apply a layout first to see your personalized dashboard!")

# Tab 5: System Information
with tab5:
    st.header("🔧 System Information")
    
    st.subheader("📊 Learning Progress")
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.metric("Total Events", len(st.session_state.user_logs))
    with col2:
        layout_applications = len([log for log in st.session_state.user_logs if log['event_type'] == 'layout_applied'])
        st.metric("Layout Applications", layout_applications)
    with col3:
        chart_clicks = len([log for log in st.session_state.user_logs if log['event_type'] == 'chart_clicked'])
        st.metric("Chart Interactions", chart_clicks)
    
    st.subheader("🧠 Model Training Data")
    if st.session_state.user_logs:
        # Show last 5 logged events
        st.write("Last 5 Events:")
        for log in st.session_state.user_logs[-5:]:
            ts = log.get('timestamp', 'N/A')
            etype = log.get('event_type', 'N/A')
            details = log.get('details', {})
            st.write(f"- {ts} — *{etype.replace('_', ' ').title()}* — {details}")
    else:
        st.info("No user events logged yet.")