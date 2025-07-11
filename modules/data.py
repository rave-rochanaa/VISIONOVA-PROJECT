import streamlit as st
import pandas as pd
import numpy as np
import json
import os

def load_data():
    try:
        df = pd.read_csv("marketing_campaign_dataset.csv")
        return preprocess_dataset(df)
    except Exception:
        return create_sample_data()

def create_sample_data():
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=100),
        "Campaign_ID": [str(i) for i in range(1, 101)],
        "Company": np.random.choice(["Innovate Industries", "Acme Corp", "Beta LLC"], 100),
        "Campaign_Type": np.random.choice(["Email", "Social Media", "PPC", "Display"], 100),
        "Target_Audience": np.random.choice(["Men 18-24", "Women 25-34", "All Adults"], 100),
        "Duration": np.random.choice(["30 days", "60 days", "90 days"], 100),
        "Channel_Used": np.random.choice(["Google Ads", "Facebook", "LinkedIn"], 100),
        "Conversion_Rate": np.random.uniform(0.01, 0.15, 100),
        "Acquisition_Cost": np.random.uniform(5000, 20000, 100),
        "ROI": np.random.uniform(0.5, 5.0, 100),
        "Location": np.random.choice(["Chicago", "New York", "Los Angeles"], 100),
        "Language": np.random.choice(["English", "Spanish", "French"], 100),
        "Clicks": np.random.randint(100, 5000, 100),
        "Impressions": np.random.randint(1000, 100000, 100),
        "Engagement_Score": np.random.uniform(1, 10, 100),
        "Customer_Segment": np.random.choice(["Health & Wellness", "Tech", "Finance"], 100)
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def preprocess_dataset(df):
    # Ensure all required columns exist
    expected_cols = [
        "Date", "Campaign_ID", "Company", "Campaign_Type", "Target_Audience", "Duration",
        "Channel_Used", "Conversion_Rate", "Acquisition_Cost", "ROI", "Location", "Language",
        "Clicks", "Impressions", "Engagement_Score", "Customer_Segment"
    ]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded dataset missing columns: {', '.join(missing_cols)}")

    # Clean 'Acquisition_Cost' remove $ and commas, convert to float
    if df["Acquisition_Cost"].dtype == object:
        df["Acquisition_Cost"] = df["Acquisition_Cost"].str.replace('[\$,]', '', regex=True)
    df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"], errors='coerce')

    # Parse Date to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Ensure numeric columns are numeric dtype
    numeric_cols = ["Conversion_Rate", "Acquisition_Cost", "ROI", "Clicks", "Impressions", "Engagement_Score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optionally convert Campaign_ID to string (if numbers)
    df["Campaign_ID"] = df["Campaign_ID"].astype(str)

    return df
def init_default_charts():
    if 'charts' not in st.session_state:
        st.session_state.charts = {}
    if 'default_charts' not in st.session_state.charts:
        st.session_state.charts['default_charts'] = {
            "chart_1": {
                "id": "chart_1",
                "type": "bar",
                "x": "Channel_Used",
                "y": "Clicks",
                "tab": "Dashboard",  # single tab name for all charts
                "position": (0, 0),  # row 0, col 0
                "title": "Clicks by Channel"
            },
            "chart_2": {
                "id": "chart_2",
                "type": "line",
                "x": "Date",
                "y": "Conversion_Rate",
                "agg": "mean",
                "tab": "Dashboard",
                "position": (0, 1),  # row 0, col 1
                "title": "Conversion Rate Over Time"
            },
            "chart_3": {
                "id": "chart_3",
                "type": "pie",
                "names": "Campaign_Type",
                "values": "Impressions",
                "tab": "Dashboard",
                "position": (1, 0),  # row 1, col 0
                "title": "Impressions by Campaign Type"
            },
            "chart_4": {
                "id": "chart_4",
                "type": "scatter",
                "x": "Clicks",
                "y": "ROI",
                "color": "Campaign_Type",
                "tab": "Dashboard",
                "position": (1, 1),  # row 1, col 1
                "title": "Clicks vs ROI by Campaign"
            },
            "chart_5": {
                "id": "chart_5",
                "type": "bar",
                "x": "Company",
                "y": "Acquisition_Cost",
                "tab": "Dashboard",
                "position": (2, 0),  # row 2, col 0
                "title": "Acquisition Cost by Company"
            },
            "chart_6": {
                "id": "chart_6",
                "type": "line",
                "x": "Date",
                "y": "Engagement_Score",
                "agg": "mean",
                "tab": "Dashboard",
                "position": (2, 1),  # row 2, col 1
                "title": "Engagement Score Over Time"
            },
            "chart_7": {
                "id": "chart_7",
                "type": "pie",
                "names": "Location",
                "values": "Clicks",
                "tab": "Dashboard",
                "position": (3, 0),  # row 3, col 0
                "title": "Clicks by Location"
            },
            "chart_8": {
                "id": "chart_8",
                "type": "bar",
                "x": "Customer_Segment",
                "y": "Conversion_Rate",
                "tab": "Dashboard",
                "position": (3, 1),  # row 3, col 1
                "title": "Conversion Rate by Customer Segment"
            }
        }
    if 'custom_charts' not in st.session_state.charts:
        st.session_state.charts['custom_charts'] = {}

def dispatch_chart(chart_config):
    pos = chart_config.get("position", (0, 0))
    if pos == (0, 0):
        plot_top_left(chart_config)
    elif pos == (0, 1):
        plot_top_right(chart_config)
    elif pos == (1, 0):
        plot_bottom_left(chart_config)
    elif pos == (1, 1):
        plot_bottom_right(chart_config)
    else:
        st.subheader(chart_config["title"])
        st.write("Chart goes here.")  # fallback\
            
def save_charts():
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/chart_config.json", "w") as f:
        json.dump(st.session_state.charts, f)

def load_charts():
    init_default_charts()
    if os.path.exists("data/chart_config.json"):
        try:
            with open("data/chart_config.json", "r") as f:
                loaded_charts = json.load(f)
                if 'default_charts' in loaded_charts:
                    st.session_state.charts['default_charts'].update(loaded_charts.get('default_charts', {}))
                if 'custom_charts' in loaded_charts:
                    st.session_state.charts['custom_charts'].update(loaded_charts.get('custom_charts', {}))
        except Exception as e:
            st.error(f"Failed loading charts config: {e}")

def get_columns_by_type(df, data_type=None):
    if data_type == 'numeric':
        return df.select_dtypes(include=['number']).columns.tolist()
    elif data_type == 'categorical':
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    elif data_type == 'datetime':
        return df.select_dtypes(include=['datetime']).columns.tolist()
    else:
        return df.columns.tolist()

def delete_chart(chart_id):
    """Delete a chart from session_state and save."""
    if 'charts' not in st.session_state:
        return

    if chart_id in st.session_state.charts.get('default_charts', {}):
        del st.session_state.charts['default_charts'][chart_id]
    elif chart_id in st.session_state.charts.get('custom_charts', {}):
        del st.session_state.charts['custom_charts'][chart_id]

    save_charts()
    st.success(f"Chart '{chart_id}' deleted successfully!")
    