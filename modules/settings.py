import streamlit as st
import pandas as pd
import re # Import regular expressions for cleaning names
import io # Import io for string handling

# Assuming modules.data exists and contains load_data, save_charts, init_default_charts
# from modules.data import load_data, save_charts, init_default_charts

# Placeholder functions for modules.data if the actual module isn't available
# In a real application, ensure these are correctly imported from your module

def load_data():
    """Loads default data or previously uploaded data."""
    if 'uploaded_data' in st.session_state:
        return st.session_state['uploaded_data']
    # Load your default dataset here if no data is uploaded
    # Example: return pd.read_csv('path/to/default_data.csv')
    st.warning("No data loaded. Please upload a dataset in Settings.")
    return pd.DataFrame() # Return empty DataFrame if no data

def save_charts():
    """Saves chart configurations (placeholder)."""
    # Placeholder function - implement actual saving logic if needed
    pass

def init_default_charts():
    """Initializes default chart configurations (placeholder)."""
    # Placeholder function - implement actual initialization logic if needed
    if 'charts' not in st.session_state:
        st.session_state['charts'] = {} # Example: Initialize an empty dict for charts

def standardize_col_name(col_name):
    """Standardizes column names for flexible matching."""
    # Strip leading/trailing whitespace, convert to lowercase, replace spaces and underscores with nothing, remove non-alphanumeric
    name = col_name.strip().lower()
    name = re.sub(r'[\s_]', '', name) # Remove spaces and underscores
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def validate_dataset(df):
    """
    Validates the dataset columns, allowing for variations in casing, spaces, and underscores.
    Renames columns in the DataFrame to match the required standard names if a match is found.
    Includes print statements for debugging.
    """
    required_cols = [
        "Campaign_ID", "Date", "Campaign_Type", "Channel_Used", "Location",
        "Company", "Target_Audience", "Conversion_Rate", "Clicks", "ROI",
        "Engagement_Score", "Impressions", "Acquisition_Cost"
    ]

    print("\n--- Debugging Column Validation ---")
    print(f"Required Columns: {required_cols}")

    # Standardize required column names
    standardized_required_cols = {standardize_col_name(col): col for col in required_cols}
    print(f"Standardized Required Columns: {standardized_required_cols}")

    # Standardize uploaded dataset column names and create a mapping to original names
    standardized_uploaded_cols_map = {}
    print("\nUploaded Columns:")
    for col in df.columns:
        std_col = standardize_col_name(col)
        standardized_uploaded_cols_map[std_col] = col
        print(f"  Original: '{col}' -> Standardized: '{std_col}'")

    missing_standardized_cols = []
    rename_map = {}

    # Check if all standardized required columns exist in the standardized uploaded columns
    for std_req_col, original_req_col in standardized_required_cols.items():
        if std_req_col in standardized_uploaded_cols_map:
            # If found, get the original uploaded column name and add to rename map
            original_uploaded_col = standardized_uploaded_cols_map[std_req_col]
            if original_uploaded_col != original_req_col:
                 rename_map[original_uploaded_col] = original_req_col
        else:
            # If not found, the original required column name is missing
            missing_standardized_cols.append(original_req_col)

    print(f"\nMissing Standardized Required Columns: {missing_standardized_cols}")
    print(f"Columns to Rename: {rename_map}")
    print("--- End Debugging Column Validation ---\n")


    if missing_standardized_cols:
        raise ValueError(f"Missing required columns (or similar variations): {', '.join(missing_standardized_cols)}")

    # If validation passes, rename columns in the DataFrame
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        st.info(f"Renamed columns to match standard schema: {rename_map}")

    return True # Validation successful

def settings_manager():
    st.header("⚙ Settings")

    st.markdown("### 1. Upload a New Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="File must contain columns similar to the marketing campaign schema")

    if uploaded_file:
        try:
            # Attempt to read the CSV directly with comma delimiter, python engine, and quote handling
            # Removed the manual line-by-line reading loop as it caused timeouts.
            df_new = pd.read_csv(uploaded_file, sep=',', engine='python', quotechar='"')

            # Validate and potentially rename columns in df_new
            validate_dataset(df_new)
            st.success("Dataset uploaded and validated successfully!")
            # Store the validated and potentially renamed dataframe
            st.session_state['uploaded_data_full'] = df_new
        except ValueError as ve:
            st.error(str(ve))
        except Exception as ex:
            st.error(f"Error processing file: {ex}")


    # If uploaded full data exists, allow customizing inputs
    if 'uploaded_data_full' in st.session_state:
        df_orig = st.session_state['uploaded_data_full']
        st.markdown("### 2. Customize Dataset Inputs")

        # Sampling slider
        sample_pct = st.slider("Select % of data to use", min_value=10, max_value=100, value=100)

        # Filter options: for demonstration, filter Location and Campaign_Type
        # Use the column names from the validated dataframe (which are now standardized)
        filter_locations = st.multiselect("Filter Locations", options=df_orig['Location'].unique(), default=list(df_orig['Location'].unique()))
        filter_campaigns = st.multiselect("Filter Campaign Types", options=df_orig['Campaign_Type'].unique(), default=list(df_orig['Campaign_Type'].unique()))

        # Apply filters and sampling
        df_filtered = df_orig[
            df_orig['Location'].isin(filter_locations) &
            df_orig['Campaign_Type'].isin(filter_campaigns)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if sample_pct < 100:
            df_filtered = df_filtered.sample(frac=sample_pct / 100, random_state=42)

        st.write(f"Dataset size after filtering and sampling: {df_filtered.shape[0]} rows")

        # Button to apply customized dataset
        if st.button("Apply Customized Dataset"):
            st.session_state['uploaded_data'] = df_filtered.reset_index(drop=True)
            init_default_charts()  # reset chart defaults if needed
            save_charts()
            st.success("Customized dataset applied! Navigate to Dashboard to see updated data.")

        # Button to clear uploaded dataset and revert
        if st.button("Clear uploaded dataset and revert to default"):
            if 'uploaded_data' in st.session_state:
                del st.session_state['uploaded_data']
            if 'uploaded_data_full' in st.session_state:
                del st.session_state['uploaded_data_full']
            init_default_charts()
            save_charts()
            st.success("Reverted to default dataset. Please refresh dashboard.")

    # Additional placeholders for other settings
    st.markdown("---")
    st.markdown("### Other Settings (Coming Soon)")
    st.text_input("Example: API Key", type="password")
    st.checkbox("Enable notifications")
    st.selectbox("Theme", options=["Light", "Dark", "System Default"])
