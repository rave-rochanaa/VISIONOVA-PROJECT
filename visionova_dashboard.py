import streamlit as st
import pandas as pd
from modules.security import check_auth
from modules.data import load_data, load_charts
from modules.dashboard import show_dashboard
from modules.chart_studio import chart_studio_page, custom_dashboard_page
from modules.nlp_interface import rag_nlp_page
from modules.prediction import prediction_engine
from modules.settings import settings_manager
from modules.regression_dashboard import classification_engine1
from modules.upload_page import upload_page1, get_uploaded_data
from modules.storytell import generate_data_story
from modules.pivort import pivot_analytics_page

# Import your AI dashboard functions
from modules.ai_dashboard import (
    ChartRecommendationEngine,
    custom_dashboard_page as ai_custom_dashboard_page,
    display_data_insights,
    display_feature_selector,
    display_chart_recommendations,
    create_advanced_dashboard_features
)

def main():
    # Uncomment below for authentication if needed
    # if not check_auth():
    #     return
    
    st.sidebar.title("VisioNova")
    app_mode = st.sidebar.radio("Mode", [
        "📊 Dashboard",
        "Custom Dashboard", 
        "🤖 Recommendation",  
        "📖 Storytelling",  # NEW MODE
        "📁 Upload Data",
        "🛠 Chart Studio",
        "💬 NLP Commands",
        "🔮 Regression",
        "📈 Classification"
    ])

    # Load data - prioritize uploaded data over default
    df = get_uploaded_data()
    if df is None:
        df = load_data()  # Fallback to default load_data function

    if 'charts' not in st.session_state or not isinstance(st.session_state.charts, dict):
        st.session_state.charts = {'default_charts': {}, 'custom_charts': {}}

    # Route to the selected app mode
    if app_mode == "📁 Upload Data":
        upload_page1()
    elif app_mode == "📊 Dashboard":
        if df is not None:
            show_dashboard(df)
        else:
            st.warning("⚠️ No data available. Please upload data first!")
            if st.button("Go to Upload Page"):
                st.session_state.app_mode = "📁 Upload Data"
                st.rerun()
    elif app_mode == "Custom Dashboard":
        if df is not None:
            custom_dashboard_page(df)
        else:
            st.warning("⚠️ No data available. Please upload data first!")
    elif app_mode == "🤖 Recommendation":
        if df is not None:
            ai_custom_dashboard_page(df)  # Call your AI dashboard function
        else:
            st.warning("⚠️ No data available. Please upload data first!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📁 Upload Data"):
                    st.session_state.app_mode = "📁 Upload Data"
                    st.rerun()
            with col2:
                if st.button("🎲 Try Sample Data"):
                    # Generate sample data for demo
                    st.session_state.sample_data = generate_sample_data()
                    st.success("Sample data loaded! Refresh to see AI recommendations.")
                    st.rerun()
    elif app_mode == "📖 Storytelling":
        if df is not None:
            st.title("📖 Data Storytelling")
            
            # Generate and display story
            with st.spinner("Analyzing your data and crafting the story..."):
                story = generate_data_story(df)
            
            if isinstance(story, str):
                st.warning(story)
            else:
                # Display the story sections
                for section in story:
                    with st.container():
                        st.markdown(f"### {section['section']}")
                        st.markdown(section['content'])
                        
                        # Display visualization if exists
                        if section['visualization']:
                            st.pyplot(section['visualization'])
                            
                        st.markdown("---")
        else:
            st.warning("⚠️ No data available. Please upload data first!")
            if st.button("📁 Upload Data"):
                st.session_state.app_mode = "📁 Upload Data"
                st.rerun()
    
    elif app_mode == "🛠 Chart Studio":
        if df is not None:
            chart_studio_page(df)
        else:
            st.warning("⚠️ No data available. Please upload data first!")
    elif app_mode == "💬 NLP Commands":
        if df is not None:
            pivot_analytics_page()
            # rag_nlp_page()
        else:
            st.warning("⚠️ No data available. Please upload data first!")
    elif app_mode == "🔮 Predictions":
        if df is not None:
            prediction_engine(df)
        else:
            st.warning("⚠️ No data available. Please upload data first!")
    elif app_mode == "📈 Regression":
        if df is not None:
            classification_engine1(df)   # Fixed function name
        else:
            st.warning("⚠️ No data available. Please upload data first!")
    elif app_mode == "⚙ Settings":
        settings_manager()

def generate_sample_data():
    """Generate sample data for AI dashboard demonstration"""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'sales': np.random.normal(1000, 300, n_samples),
        'profit': np.random.normal(200, 100, n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'customer_satisfaction': np.random.uniform(1, 5, n_samples),
        'marketing_spend': np.random.exponential(100, n_samples)
    }
    
    df = pd.DataFrame(data)
    # Add some correlations
    df['profit'] = df['sales'] * 0.3 + np.random.normal(0, 50, n_samples)
    df['customer_satisfaction'] = 3 + (df['sales'] - 1000) / 1000 + np.random.normal(0, 0.5, n_samples)
    df['customer_satisfaction'] = np.clip(df['customer_satisfaction'], 1, 5)
    
    return df

# Enhanced data loading to include sample data
def enhanced_get_data():
    """Enhanced data loading with sample data fallback"""
    # First try uploaded data
    df = get_uploaded_data()
    if df is not None:
        return df
    
    # Then try session state sample data
    if 'sample_data' in st.session_state:
        return st.session_state.sample_data
    
    # Finally try default load_data
    try:
        df = load_data()
        return df
    except:
        return None

# Update your data loading in main function
def main_enhanced():
    """Enhanced main function with better data handling"""
    # Uncomment below for authentication if needed
    # if not check_auth():
    #     return
    
    st.sidebar.title("VisioNova")
    
    # Add data status indicator
    df = enhanced_get_data()
    if df is not None:
        st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    else:
        st.sidebar.warning("⚠️ No data available")
    
    app_mode = st.sidebar.radio("Mode", [
        "📊 Dashboard",
        "Custom Dashboard", 
        "🤖 Chart Recommendation",  # AI-powered dashboard
        "📖 Storytelling",
        "📁 Upload Data",
        "🛠 Chart Studio",
        "💬 Chat with data",
        "🔮 Regression",
        "📈 Classification"
    ])

    if 'charts' not in st.session_state or not isinstance(st.session_state.charts, dict):
        st.session_state.charts = {'default_charts': {}, 'custom_charts': {}}

    # Data availability check helper
    def require_data(func, *args, **kwargs):
        if df is not None:
            return func(df, *args, **kwargs)
        else:
            st.warning("⚠️ No data available. Please upload data first!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📁 Upload Data"):
                    return upload_page1()
            with col2:
                if st.button("🎲 Generate Sample Data"):
                    st.session_state.sample_data = generate_sample_data()
                    st.success("Sample data generated! Select a mode to continue.")
                    st.rerun()

    # Route to the selected app mode
    if app_mode == "📁 Upload Data":
        upload_page1()
    elif app_mode == "📊 Dashboard":
        require_data(show_dashboard)
    elif app_mode == "Custom Dashboard":
        require_data(custom_dashboard_page)
    elif app_mode == "🤖 Chart Recommendation":
        require_data(ai_custom_dashboard_page)  # Your AI dashboard
    elif app_mode == "📖 Storytelling":
        if df is not None:
            st.title("📖 Data Storytelling")
            
            # Generate and display story
            with st.spinner("Analyzing your data and crafting the story..."):
                story = generate_data_story(df)
            
            if isinstance(story, str):
                st.warning(story)
            else:
                # Display the story sections
                for section in story:
                    with st.container():
                        st.markdown(f"### {section['section']}")
                        st.markdown(section['content'])
                        
                        # Display visualization if exists
                        if section['visualization']:
                            st.pyplot(section['visualization'])
                            
                        st.markdown("---")
        else:
            st.warning("⚠️ No data available. Please upload data first!")
            if st.button("📁 Upload Data"):
                st.session_state.app_mode = "📁 Upload Data"
                st.rerun()
    elif app_mode == "🛠 Chart Studio":
        require_data(chart_studio_page)
    elif app_mode == "💬 Chat with data":
        require_data(lambda df: pivot_analytics_page())  # NLP doesn't need df parameter
    elif app_mode == "🔮 Regression":
        require_data(prediction_engine)
    elif app_mode == "📈 Classification":
        require_data(classification_engine1)


# Load charts and run the main function
load_charts()

if __name__ == "__main__":
    main_enhanced()  # Use the enhanced version
else:
    main_enhanced()  # For streamlit run