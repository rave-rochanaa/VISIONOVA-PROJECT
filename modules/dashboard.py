# DASHBOARD.PY - ENHANCED VERSION WITH CREATIVE FEATURES
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from modules.data import get_columns_by_type, save_charts, delete_chart, preprocess_dataset
from modules.chart_studio import render_chart, CHART_COMPATIBILITY
from modules.prediction import prediction_engine
from modules.settings import settings_manager
from modules.regression_dashboard import classification_engine1 

# --- Enhanced Corporate Styling with Animations ---
st.set_page_config(page_title="US Campaign Performance Dashboard", layout="wide")

st.markdown("""
    <style>
    :root {
        --primary: #2E5BFF;
        --secondary: #8C98FF;
        --background: #F8F9FC;
        --card-bg: #FFFFFF;
        --text-primary: #2E384D;
        --text-secondary: #6B7A99;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
    }
    
    body {
        background-color: var(--background);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .header {
        font-size: 24px !important;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateY(-10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 1px 3px rgba(28,41,90,0.08); }
        50% { box-shadow: 0 4px 12px rgba(46,91,255,0.15); }
    }
    
    .compact-metric {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 12px;
        margin: 2px;
        box-shadow: 0 1px 3px rgba(28,41,90,0.08);
        border: 1px solid #E0E7FF;
        text-align: center;
        min-height: 60px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .compact-metric:hover {
        transform: translateY(-2px);
        animation: glow 1.5s ease-in-out infinite;
        cursor: pointer;
    }
    
    .compact-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .compact-metric-title {
        font-size: 11px;
        color: var(--text-secondary);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        font-weight: 500;
    }
    
    .compact-metric-value {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    
    .quick-stats-container {
        background: linear-gradient(135deg, #F0F3FF 0%, #E8EDFF 100%);
        border-radius: 8px;
        padding: 12px;
        border: 2px solid var(--primary);
        text-align: center;
        animation: slideIn 0.5s ease-out;
        position: relative;
    }
    
    .quick-stats-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .quick-stats-value {
        font-size: 16px;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .quick-stats-trend {
        font-size: 10px;
        margin-top: 4px;
        font-weight: 500;
    }
    
    .trend-up { color: var(--success); }
    .trend-down { color: var(--danger); }
    .trend-neutral { color: var(--warning); }
    
    .chart-container {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #E0E7FF;
        height: 100%;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .chart-container:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(46, 91, 255, 0.1);
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .chart-container:hover::before {
        opacity: 1;
    }
    
    .chart-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
        text-align: center;
        position: relative;
    }
    
    .compact-filter {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #E0E7FF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .top-campaign {
        background: linear-gradient(135deg, #F0F3FF 0%, #E8EDFF 100%);
        border-radius: 6px;
        padding: 8px 10px;
        margin: 4px 0;
        border-left: 3px solid var(--primary);
        font-size: 11px;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .top-campaign:hover {
        transform: translateX(4px);
        box-shadow: 0 2px 8px rgba(46, 91, 255, 0.15);
    }
    
    .top-campaign-title {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 10px;
    }
    
    .top-campaign-metric {
        color: var(--text-secondary);
        font-size: 9px;
        margin-top: 2px;
    }
    
    .performance-badge {
        position: absolute;
        top: 8px;
        right: 8px;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 8px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-excellent { background: var(--success); color: white; }
    .badge-good { background: var(--warning); color: white; }
    .badge-poor { background: var(--danger); color: white; }
    
    .stSelectbox > div > div {
        font-size: 12px;
    }
    .stMultiSelect > div > div {
        font-size: 12px;
    }
    .stDateInput > div > div {
        font-size: 12px;
    }
    
    /* Loading animation */
    .loading-pulse {
        animation: loadingPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes loadingPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Tooltip styles */
    .metric-tooltip {
        position: relative;
        cursor: help;
    }
    
    .metric-tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--text-primary);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 10px;
        white-space: nowrap;
        z-index: 1000;
    }
    </style>
""", unsafe_allow_html=True)

def get_performance_badge(roi):
    """Get performance badge based on ROI"""
    if roi >= 7:
        return "badge-excellent", "⭐ EXCELLENT"
    elif roi >= 2:
        return "badge-good", "👍 GOOD"
    else:
        return "badge-poor", "⚠️ NEEDS IMPROVEMENT"
    
def calculate_trend(current, previous):
    """Calculate trend percentage with bounds (0-100%)"""
    if previous == 0:
        return 0, "neutral"
    
    # Handle cases where current or previous might be negative
    if (current < 0) or (previous < 0):
        return 0, "neutral"
    
    # Calculate percentage change with bounds
    try:
        change = ((current - previous) / previous) * 100
        change = max(-100, min(100, change))  # Bound between -100% and +100%
    except:
        change = 0
    
    # Determine trend direction
    if change > 5:
        return change, "up"
    elif change < -5:
        return change, "down"
    else:
        return change, "neutral"
    
def create_enhanced_metric_card(title, value, icon, color, tooltip="", trend_value=None):
    """Create enhanced metric card with animations and trends"""
    trend_html = ""
    if trend_value is not None:
        try:
            # Convert value to float for calculation
            current_val = float(str(value).replace('K', '').replace('$', '').replace('%', '').replace(',', ''))
        except:
            current_val = 0
        
        trend_change, trend_direction = calculate_trend(current_val, trend_value)
        trend_class = f"trend-{trend_direction}"
        trend_symbol = "▲" if trend_direction == "up" else "▼" if trend_direction == "down" else "●"
        trend_html = f"""<div class='quick-stats-trend {trend_class}'>{trend_symbol} {abs(trend_change):.1f}%</div>"""
    
    return f"""
        <div class='compact-metric metric-tooltip' data-tooltip='{tooltip}'>
            <div class='compact-metric-title'>{icon} {title}</div>
            <div class='compact-metric-value' style='color: {color}'>{value}</div>
            {trend_html}
        </div>
    """


def create_compact_map(data):
    state_mapping = {
        'Chicago': 'IL', 'Houston': 'TX', 'Los Angeles': 'CA',
        'Miami': 'FL', 'New York': 'NY'
    }
    
    state_df = data[data['Location'].isin(state_mapping.keys())].copy()
    state_df['State'] = state_df['Location'].map(state_mapping)
    state_agg = state_df.groupby('State').agg(
        Average_ROI=('ROI', 'mean'),
        Total_Campaigns=('Campaign_ID', 'nunique')
    ).reset_index()

    # FIXED: Complete state list
    all_states = pd.DataFrame({'State': [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]})

    merged_df = all_states.merge(state_agg, on='State', how='left').fillna(0)

    fig = px.choropleth(
        merged_df,
        locations='State',
        locationmode='USA-states',
        color='Average_ROI',
        color_continuous_scale='Viridis',
        scope='usa',
        range_color=(0, merged_df['Average_ROI'].max()),
        hover_data={'State': True, 'Average_ROI': ':.2f', 'Total_Campaigns': True},
        labels={'Average_ROI': 'ROI'}
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title='', thickness=8, len=0.4),
        dragmode=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        font=dict(size=10)
    )
    return fig

def create_compact_chart(df, chart_type, x_col, y_col, color_col=None, height=180):
    """Create compact charts with enhanced styling and animations"""
    if chart_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, height=height,
                    color_discrete_sequence=px.colors.qualitative.Set3)
    elif chart_type == 'line':
        fig = px.line(df, x=x_col, y=y_col, height=height)
        fig.update_traces(line=dict(width=3))
    elif chart_type == 'scatter':
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, height=height,
                        color_discrete_sequence=px.colors.qualitative.Set3)
    elif chart_type == 'pie':
        fig = px.pie(df, names=x_col, values=y_col, height=height,
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=9),
        showlegend=False if chart_type != 'pie' else True,
        legend=dict(font=dict(size=8)) if chart_type == 'pie' else None
    )
    
    if chart_type != 'pie':
        fig.update_xaxes(tickfont=dict(size=8), gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(tickfont=dict(size=8), gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for KPIs"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 12}},
        delta = {'reference': max_value * 0.7},
        gauge = {
            'axis': {'range': [None, max_value], 'tickfont': {'size': 8}},
            'bar': {'color': "#2E5BFF"},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "lightgray"},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=8)
    )
    return fig

def show_dashboard(df):
    # Header with enhanced styling
    st.markdown("<div class='header'>📊 US Campaign Performance Dashboard</div>", unsafe_allow_html=True)

    # --- Enhanced Filters in single row ---
    with st.container():
        st.markdown("<div class='compact-filter'>", unsafe_allow_html=True)
        cols = st.columns(5)
        with cols[0]:
            location = st.multiselect("📍 Location", sorted(df["Location"].dropna().unique()), key="loc")
        with cols[1]:
            channel = st.multiselect("📺 Channel", sorted(df["Channel_Used"].dropna().unique()), key="ch")
        with cols[2]:
            company = st.multiselect("🏢 Company", sorted(df["Company"].dropna().unique()), key="comp")
        with cols[3]:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            min_date = df["Date"].dropna().min()
            max_date = df["Date"].dropna().max()
            if pd.isna(min_date) or pd.isna(max_date):
                today = pd.to_datetime("today").date()
                min_date = max_date = today
            else:
                min_date = min_date.date()
                max_date = max_date.date()
            date_range = st.date_input("📅 Date Range", value=(min_date, max_date), key="date")
        with cols[4]:
            # Fixed Quick Stats with proper styling
            total_records = len(df)
            unique_companies = df["Company"].nunique()
            avg_performance = df["ROI"].mean()
            
            st.markdown(f"""
                <div class='quick-stats-container'>
                    <div class='quick-stats-title'>📊 Quick Stats</div>
                    <div class='quick-stats-value'>{total_records:,} Records</div>
                    <div class='quick-stats-value'>{unique_companies} Companies</div>
                    <div class='quick-stats-trend trend-{"up" if avg_performance > 5 else "neutral"}'>
                        Avg Performance: {avg_performance:.1f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Data Filtering ---
    filtered_df = df.copy()
    if location:
        filtered_df = filtered_df[filtered_df["Location"].isin(location)]
    if channel:
        filtered_df = filtered_df[filtered_df["Channel_Used"].isin(channel)]
    if company:
        filtered_df = filtered_df[filtered_df["Company"].isin(company)]

    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], errors='coerce')
    if date_range:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
            (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
        ]

    # --- Enhanced KPI Row with animations and trends ---
    cols = st.columns(6)
    
    # Calculate previous period for trends (mock data for demo)
    prev_campaigns = filtered_df["Campaign_ID"].nunique() * 0.9
    prev_roi = filtered_df['ROI'].mean() * 0.5
    prev_clicks = filtered_df['Clicks'].sum() * 0.05
    prev_cost = filtered_df['Acquisition_Cost'].mean() * 0.1
    prev_conv = filtered_df['Conversion_Rate'].mean() * 0.03
    prev_engagement = filtered_df['Engagement_Score'].mean() * 0.50
    
    kpi_data = [
        ("Campaigns", filtered_df["Campaign_ID"].nunique(), "#2E5BFF", "🎯", 
         "Total active campaigns", prev_campaigns),
        ("Avg ROI", f"{filtered_df['ROI'].mean():.1f}", "#00C1D4", "💰", 
         "Return on Investment", prev_roi),
        ("Total Clicks", f"{filtered_df['Clicks'].sum()/1000:.1f}K", "#FF7A45", "👆", 
         "User engagement clicks", prev_clicks/1000),
        ("Avg Cost", f"${filtered_df['Acquisition_Cost'].mean():.0f}", "#34AA44", "💵", 
         "Customer acquisition cost", prev_cost),
        ("Conv Rate", f"{filtered_df['Conversion_Rate'].mean():.1f}%", "#8B5CF6", "📈", 
         "Conversion percentage", prev_conv),
        ("Engagement", f"{filtered_df['Engagement_Score'].mean():.1f}", "#F59E0B", "⭐", 
         "User engagement score", prev_engagement)
    ]
    
    for (title, value, color, icon, tooltip, prev_val), col in zip(kpi_data, cols):
        with col:
            current_val = float(str(value).replace('K', '').replace('$', '').replace('%', ''))
            st.markdown(create_enhanced_metric_card(title, value, icon, color, tooltip, prev_val), 
                       unsafe_allow_html=True)

    # --- Main Dashboard Grid (3 rows) ---
    


    # ROW 1: Performance Charts with Gauges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='chart-container'><div class='chart-title'>📈 Conversion Trends</div>", unsafe_allow_html=True)
        time_df = filtered_df.groupby(pd.Grouper(key='Date', freq='W'))['Conversion_Rate'].mean().reset_index()
        fig = create_compact_chart(time_df, 'line', 'Date', 'Conversion_Rate')
        st.plotly_chart(fig, use_container_width=True, key="conv_time")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'><div class='chart-title'>💰 Cost Analysis</div>", unsafe_allow_html=True)
        cost_data = filtered_df.groupby('Company')['Acquisition_Cost'].mean().reset_index()
        fig = create_compact_chart(cost_data, 'bar', 'Company', 'Acquisition_Cost', 'Company')
        st.plotly_chart(fig, use_container_width=True, key="cost_comp")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='chart-container'><div class='chart-title'>⭐ Engagement Performance</div>", unsafe_allow_html=True)
        # Create a gauge chart for average engagement
        avg_engagement = filtered_df['Engagement_Score'].mean()
        gauge_fig = create_gauge_chart(avg_engagement, "Engagement", 10)
        st.plotly_chart(gauge_fig, use_container_width=True, key="engagement_gauge")
        st.markdown("</div>", unsafe_allow_html=True)
    

    # ROW 2: Advanced Analytics with Enhanced Scatter Plot
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='chart-container'><div class='chart-title'>🎯 Performance Matrix: Clicks vs ROI</div>", unsafe_allow_html=True)
        fig = px.scatter(
            filtered_df, 
            x='Clicks', 
            y='ROI', 
            color='Campaign_Type',
            size='Impressions',
            hover_name='Campaign_ID',
            hover_data=['Company', 'Conversion_Rate', 'Acquisition_Cost'],
            height=180,
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=9),
            legend=dict(font=dict(size=8), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Add trend line
        fig.add_traces(px.scatter(filtered_df, x='Clicks', y='ROI', trendline="ols", opacity=0).data[1:])
        st.plotly_chart(fig, use_container_width=True, key="scatter_perf")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Campaign Type Distribution with enhanced styling
        st.markdown("<div class='chart-container'><div class='chart-title'>📊 Campaign Distribution</div>", unsafe_allow_html=True)
        type_data = filtered_df['Campaign_Type'].value_counts().reset_index()
        type_data.columns = ['Campaign_Type', 'Count']
        fig = create_compact_chart(type_data, 'pie', 'Campaign_Type', 'Count', height=80)
        st.plotly_chart(fig, use_container_width=True, key="type_dist")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ROI Performance Gauge
        st.markdown("<div class='chart-container'><div class='chart-title'>🎯 ROI Performance</div>", unsafe_allow_html=True)
        avg_roi = filtered_df['ROI'].mean()
        roi_gauge = create_gauge_chart(avg_roi, "ROI Score", 15)
        st.plotly_chart(roi_gauge, use_container_width=True, key="roi_gauge")
        st.markdown("</div>", unsafe_allow_html=True)
        # ROW 1: Map + Top Campaigns + Channel Performance
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<div class='chart-container'><div class='chart-title'>🗺️ Regional Performance Heatmap</div>", unsafe_allow_html=True)
        map_fig = create_compact_map(filtered_df)
        st.plotly_chart(map_fig, use_container_width=True, key="map")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'><div class='chart-title'>🏆 Top 5 Performing Campaigns</div>", unsafe_allow_html=True)
        top_campaigns = filtered_df.sort_values("ROI", ascending=False).head(5)
        for idx, (_, row) in enumerate(top_campaigns.iterrows()):
            badge_class, badge_text = get_performance_badge(row['ROI'])
            st.markdown(f"""
                <div class='top-campaign'>
                    <div class='performance-badge {badge_class}'>{badge_text}</div>
                    <div class='top-campaign-title'>{row['Company'][:15]}</div>
                    <div class='top-campaign-metric'>ROI: {row['ROI']:.1f} | Cost: ${row['Acquisition_Cost']:,.0f}</div>
                    <div class='top-campaign-metric'>Conv: {row['Conversion_Rate']:.1f}% | Clicks: {row['Clicks']:,}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='chart-container'><div class='chart-title'>📺 Channel Distribution</div>", unsafe_allow_html=True)
        channel_data = filtered_df.groupby('Channel_Used')['Clicks'].sum().reset_index()
        fig = create_compact_chart(channel_data, 'pie', 'Channel_Used', 'Clicks', height=160)
        st.plotly_chart(fig, use_container_width=True, key="channel_pie")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Enhanced Data Table with search and sorting ---
    with st.expander("📋 Detailed Campaign Analytics", expanded=False):
        st.markdown("### 🔍 Interactive Campaign Data")
        search_term = st.text_input("🔎 Search campaigns", placeholder="Enter campaign ID, company, or location...")
        
        display_cols = ['Campaign_ID', 'Company', 'Channel_Used', 'Location', 'Campaign_Type', 
                       'Clicks', 'Impressions', 'Conversion_Rate', 'ROI', 'Acquisition_Cost', 'Engagement_Score']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        display_df = filtered_df[available_cols].copy()
        if search_term:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]
        
        # FIX: Check if ROI exists before adding performance column
        if 'ROI' in display_df.columns:
            display_df['Performance'] = display_df['ROI'].apply(
                lambda x: "🟢 Excellent" if x >= 8 else "🟡 Good" if x >= 3 else "🔴 Needs Improvement"
            )
        
        # Sort options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("📊 Sort by", available_cols, index=available_cols.index('ROI') if 'ROI' in available_cols else 0)
        with col2:
            sort_order = st.selectbox("📈 Order", ["Descending", "Ascending"])
        
        # Apply sorting
        ascending = sort_order == "Ascending"
        display_df = display_df.sort_values(sort_by, ascending=ascending)
        
        # Display with enhanced styling
        st.dataframe(
            display_df,
            height=300,
            use_container_width=True,
            column_config={
                "ROI": st.column_config.NumberColumn(
                    "ROI",
                    help="Return on Investment",
                    format="%.2f"
                ),
                "Conversion_Rate": st.column_config.NumberColumn(
                    "Conv Rate %",
                    help="Conversion Rate Percentage",
                    format="%.2f%%"
                ),
                "Acquisition_Cost": st.column_config.NumberColumn(
                    "Cost $",
                    help="Customer Acquisition Cost",
                    format="$%.0f"
                ),
                "Clicks": st.column_config.NumberColumn(
                    "Clicks",
                    help="Total Clicks",
                    format="%d"
                ),
                "Impressions": st.column_config.NumberColumn(
                    "Impressions",
                    help="Total Impressions",
                    format="%d"
                ),
                "Engagement_Score": st.column_config.NumberColumn(
                    "Engagement",
                    help="Engagement Score (1-10)",
                    format="%.1f"
                ),
                "Performance": st.column_config.TextColumn(
                    "Performance",
                    help="Overall Performance Rating"
                )
            }
        )
        
        # Summary statistics
        if not display_df.empty:
            st.markdown("### 📊 Summary Statistics")
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric("📈 Avg ROI", f"{display_df['ROI'].mean():.2f}" if 'ROI' in display_df.columns else "N/A")
            with summary_cols[1]:
                st.metric("💰 Avg Cost", f"${display_df['Acquisition_Cost'].mean():.0f}" if 'Acquisition_Cost' in display_df.columns else "N/A")
            with summary_cols[2]:
                st.metric("🎯 Avg Conv Rate", f"{display_df['Conversion_Rate'].mean():.2f}%" if 'Conversion_Rate' in display_df.columns else "N/A")
            with summary_cols[3]:
                st.metric("⭐ Avg Engagement", f"{display_df['Engagement_Score'].mean():.2f}" if 'Engagement_Score' in display_df.columns else "N/A")

    # --- Creative Features Section ---
    st.markdown("---")
    st.markdown("### 🚀 Advanced Analytics & Insights")
    
    # Creative Feature 1: Performance Alerts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='chart-container'>
                <div class='chart-title'>🚨 Performance Alerts</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Check for performance issues
        low_roi_campaigns = filtered_df[filtered_df['ROI'] < 3]
        high_cost_campaigns = filtered_df[filtered_df['Acquisition_Cost'] > filtered_df['Acquisition_Cost'].quantile(0.9)]
        low_conversion_campaigns = filtered_df[filtered_df['Conversion_Rate'] < 2]
        
        alert_count = 0
        if not low_roi_campaigns.empty:
            st.warning(f"⚠️ {len(low_roi_campaigns)} campaigns with ROI < 3.0")
            alert_count += 1
        if not high_cost_campaigns.empty:
            st.error(f"🔴 {len(high_cost_campaigns)} campaigns with high acquisition costs")
            alert_count += 1
        if not low_conversion_campaigns.empty:
            st.info(f"📉 {len(low_conversion_campaigns)} campaigns with low conversion rates")
            alert_count += 1
        
        if alert_count == 0:
            st.success("✅ All campaigns performing well!")
        
        # Performance score calculation
        avg_roi = filtered_df['ROI'].mean()
        avg_conv = filtered_df['Conversion_Rate'].mean()
        avg_cost_efficiency = 1 / (filtered_df['Acquisition_Cost'].mean() / 100)  # Normalize cost efficiency
        
        performance_score = (avg_roi * 0.4 + avg_conv * 0.3 + avg_cost_efficiency * 0.3)
        
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 8px; color: white; margin-top: 10px;'>
            <h4>Overall Performance Score</h4>
            <h2>{performance_score:.1f}/10</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='chart-container'>
                <div class='chart-title'>🎯 Smart Recommendations</div>
            </div>
        """, unsafe_allow_html=True)
        
        recommendations = []
        
        # Generate smart recommendations
        best_channel = filtered_df.groupby('Channel_Used')['ROI'].mean().idxmax()
        worst_channel = filtered_df.groupby('Channel_Used')['ROI'].mean().idxmin()
        
        recommendations.append(f"💡 Focus on **{best_channel}** channel (highest ROI)")
        recommendations.append(f"⚠️ Review **{worst_channel}** channel strategy")
        
        best_location = filtered_df.groupby('Location')['Conversion_Rate'].mean().idxmax()
        recommendations.append(f"📍 Expand in **{best_location}** (best conversion)")
        
        if filtered_df['Acquisition_Cost'].std() > filtered_df['Acquisition_Cost'].mean() * 0.3:
            recommendations.append("💰 Standardize cost management across campaigns")
        
        high_engagement_segment = filtered_df.groupby('Customer_Segment')['Engagement_Score'].mean().idxmax()
        recommendations.append(f"🎯 Target **{high_engagement_segment}** segment more")
        
        for i, rec in enumerate(recommendations[:5], 1):
            st.markdown(f"**{i}.** {rec}")
    
    with col3:
        st.markdown("""
            <div class='chart-container'>
                <div class='chart-title'>🏅 Campaign Rankings</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create a composite score for ranking
        filtered_df['Composite_Score'] = (
            filtered_df['ROI'] * 0.3 +
            filtered_df['Conversion_Rate'] * 0.25 +
            filtered_df['Engagement_Score'] * 0.2 +
            (10 - filtered_df['Acquisition_Cost'] / filtered_df['Acquisition_Cost'].max() * 10) * 0.25
        )
        
        top_campaigns_ranked = filtered_df.nlargest(5, 'Composite_Score')
        
        for idx, (_, campaign) in enumerate(top_campaigns_ranked.iterrows(), 1):
            medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}️⃣"
            
            st.markdown(f"""
            <div style='background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); 
                        padding: 8px; margin: 4px 0; border-radius: 6px; color: white;'>
                <strong>{medal} {campaign['Company']}</strong><br>
                <small>Score: {campaign['Composite_Score']:.2f} | ROI: {campaign['ROI']:.1f}</small>
            </div>
            """, unsafe_allow_html=True)

    # Creative Feature 2: Real-time Insights Dashboard
    st.markdown("---")
    insight_cols = st.columns(2)
    
    with insight_cols[0]:
        st.markdown("### 📊 Market Intelligence")
        
        # Market share analysis
        company_performance = filtered_df.groupby('Company').agg({
            'ROI': 'mean',
            'Clicks': 'sum',
            'Campaign_ID': 'nunique'
        }).round(2)
        
        company_performance['Market_Share'] = (company_performance['Clicks'] / company_performance['Clicks'].sum() * 100).round(1)
        company_performance = company_performance.sort_values('Market_Share', ascending=False)
        
        st.markdown("**Market Share by Clicks:**")
        for company, row in company_performance.head(3).iterrows():
            st.markdown(f"• **{company}**: {row['Market_Share']}% ({row['Campaign_ID']} campaigns)")
        
        # Trend analysis
        st.markdown("**Performance Trends:**")
        if len(filtered_df) > 10:
            recent_roi = filtered_df.tail(len(filtered_df)//2)['ROI'].mean()
            earlier_roi = filtered_df.head(len(filtered_df)//2)['ROI'].mean()
            trend = "📈 Improving" if recent_roi > earlier_roi else "📉 Declining" if recent_roi < earlier_roi else "➡️ Stable"
            st.markdown(f"• ROI Trend: {trend}")
    
    with insight_cols[1]:
        st.markdown("### 🎨 Visualization Controls")
        
        # Interactive chart customization
        chart_type = st.selectbox("📊 Choose Chart Type", ["Scatter", "Heatmap", "3D Scatter", "Correlation Matrix"])
        
        if chart_type == "Heatmap":
            # Create correlation heatmap
            numeric_cols = ['ROI', 'Clicks', 'Impressions', 'Conversion_Rate', 'Acquisition_Cost', 'Engagement_Score']
            available_numeric = [col for col in numeric_cols if col in filtered_df.columns]
            
            if len(available_numeric) >= 2:
                corr_matrix = filtered_df[available_numeric].corr()
                fig = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale="RdBu",
                               title="Campaign Metrics Correlation")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "3D Scatter":
            # 3D scatter plot
            if all(col in filtered_df.columns for col in ['Clicks', 'ROI', 'Conversion_Rate']):
                fig = px.scatter_3d(filtered_df, 
                                  x='Clicks', 
                                  y='ROI', 
                                  z='Conversion_Rate',
                                  color='Campaign_Type',
                                  size='Impressions',
                                  hover_name='Campaign_ID',
                                  title="3D Performance Analysis")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# # Load your data here
# if __name__ == "__main__":
#     try:
#         df = pd.read_csv("marketing_campaign_dataset.csv")
#         df = preprocess_dataset(df)
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         from modules.data import create_sample_data
#         df = create_sample_data()
    
#     show_dashboard(df) 