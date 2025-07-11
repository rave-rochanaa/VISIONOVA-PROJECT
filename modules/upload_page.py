import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go

def upload_page1():
    """
    Comprehensive upload page with data preprocessing options
    """
    st.title("📁 Data Upload & Preprocessing")
    st.markdown("---")
    
    # File upload section
    st.header("1. Data Source Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file for analysis"
        )
    
    with col2:
        st.subheader("Or Use Sample Data")
        use_sample = st.button("Use Marketing Campaign Dataset", 
                              type="secondary",
                              help="Load the default marketing_campaign_dataset.csv")
    
    # Initialize dataframe
    df = None
    
    # Handle file upload or sample data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            st.session_state.uploaded_data = df
            st.session_state.data_source = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    elif use_sample:
        try:
            df = pd.read_csv("marketing_campaign_dataset.csv")
            st.success(f"✅ Sample data loaded successfully! Shape: {df.shape}")
            st.session_state.uploaded_data = df
            st.session_state.data_source = "marketing_campaign_dataset.csv"
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            st.info("Make sure 'marketing_campaign_dataset.csv' exists in your project directory")
    
    # Check if we have data from session state
    elif 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        st.info(f"📊 Using previously loaded data: {st.session_state.get('data_source', 'Unknown')}")
    
    if df is not None:
        # Data preview section
        st.markdown("---")
        st.header("2. Data Preview")
        
        # Basic info tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔍 Sample Data", "📊 Statistics", "❓ Missing Values"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            st.subheader("First 10 Rows")
            # Convert datetime columns to string for display
            display_df = df.head(10).copy()
            for col in display_df.select_dtypes(include=['datetime64']).columns:
                display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df, use_container_width=True)
            
            st.subheader("Last 5 Rows")
            display_df = df.tail(5).copy()
            for col in display_df.select_dtypes(include=['datetime64']).columns:
                display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df, use_container_width=True)
        
        with tab3:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(include='all'), use_container_width=True)
        
        with tab4:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                st.subheader("Missing Values Analysis")
                
                # Missing values chart
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column",
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Missing values table
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("🎉 No missing values found in the dataset!")
        
        # Data preprocessing section
        st.markdown("---")
        st.header("3. Data Preprocessing")
        
        # Create preprocessing tabs
        preprocess_tabs = st.tabs([
            "🧹 Handle Missing Values",
            "💰 Currency Conversion",
            "🔄 Data Type Conversion",
            "✂️ Text Processing",
            "📊 Feature Engineering"
        ])
        
        # Initialize processed dataframe
        if 'processed_df' not in st.session_state:
            st.session_state.processed_df = df.copy()
        
        processed_df = st.session_state.processed_df
        
        with preprocess_tabs[0]:  # Handle Missing Values
            st.subheader("Missing Values Treatment")
            
            missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
            
            if missing_cols:
                for col in missing_cols:
                    st.write(f"Column: {col} (Missing: {processed_df[col].isnull().sum()})")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if processed_df[col].dtype in ['int64', 'float64']:
                            strategy = st.selectbox(
                                "Strategy",
                                ["Keep as is", "Fill with Mean", "Fill with Median", "Fill with Mode", "Drop rows", "Fill with 0"],
                                key=f"strategy_{col}"
                            )
                        else:
                            strategy = st.selectbox(
                                "Strategy",
                                ["Keep as is", "Fill with Mode", "Fill with 'Unknown'", "Drop rows"],
                                key=f"strategy_{col}"
                            )
                    
                    with col2:
                        if st.button(f"Apply to {col}", key=f"apply_{col}"):
                            if strategy == "Fill with Mean" and processed_df[col].dtype in ['int64', 'float64']:
                                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                                st.success(f"Filled {col} with mean")
                            elif strategy == "Fill with Median" and processed_df[col].dtype in ['int64', 'float64']:
                                processed_df[col].fillna(processed_df[col].median(), inplace=True)
                                st.success(f"Filled {col} with median")
                            elif strategy == "Fill with Mode":
                                mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 0
                                processed_df[col].fillna(mode_val, inplace=True)
                                st.success(f"Filled {col} with mode")
                            elif strategy == "Fill with 0":
                                processed_df[col].fillna(0, inplace=True)
                                st.success(f"Filled {col} with 0")
                            elif strategy == "Fill with 'Unknown'":
                                processed_df[col].fillna('Unknown', inplace=True)
                                st.success(f"Filled {col} with 'Unknown'")
                            elif strategy == "Drop rows":
                                processed_df.dropna(subset=[col], inplace=True)
                                st.success(f"Dropped rows with missing {col}")
                            
                            st.session_state.processed_df = processed_df
                            st.rerun()
                
                # Bulk operations
                st.subheader("Bulk Operations")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Fill All Numeric with Mean"):
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                        st.session_state.processed_df = processed_df
                        st.success("All numeric columns filled with mean")
                        st.rerun()
                
                with col2:
                    if st.button("Fill All Categorical with Mode"):
                        categorical_cols = processed_df.select_dtypes(include=['object']).columns
                        for col in categorical_cols:
                            if processed_df[col].isnull().any():
                                mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'Unknown'
                                processed_df[col].fillna(mode_val, inplace=True)
                        st.session_state.processed_df = processed_df
                        st.success("All categorical columns filled with mode")
                        st.rerun()
                
                with col3:
                    if st.button("Drop All Rows with Missing Values"):
                        processed_df.dropna(inplace=True)
                        st.session_state.processed_df = processed_df
                        st.success("Dropped all rows with missing values")
                        st.rerun()
            else:
                st.success("No missing values to handle!")
        
        with preprocess_tabs[1]:  # Currency Conversion
            st.subheader("Currency Conversion")
            
            # Detect potential currency columns
            currency_patterns = [r'\$', r'USD', r'EUR', r'GBP', r'₹', r'¥', r'€']
            potential_currency_cols = []
            
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    sample_values = processed_df[col].dropna().astype(str).head(10)
                    for pattern in currency_patterns:
                        if any(re.search(pattern, str(val)) for val in sample_values):
                            potential_currency_cols.append(col)
                            break
            
            if potential_currency_cols:
                st.write("Detected potential currency columns:")
                for col in potential_currency_cols:
                    st.write(f"- {col}: {processed_df[col].head(3).tolist()}")
                
                selected_currency_cols = st.multiselect(
                    "Select columns to convert to numeric:",
                    potential_currency_cols,
                    default=potential_currency_cols
                )
                
                if st.button("Convert Selected Columns"):
                    for col in selected_currency_cols:
                        # Remove currency symbols and convert to float
                        processed_df[f"{col}_numeric"] = processed_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        processed_df[f"{col}_numeric"] = pd.to_numeric(processed_df[f"{col}_numeric"], errors='coerce')
                        st.success(f"Created numeric version: {col}_numeric")
                    
                    st.session_state.processed_df = processed_df
                    st.rerun()
            else:
                st.info("No obvious currency columns detected. You can manually select columns below.")
            
            # Manual currency conversion
            st.subheader("Manual Currency Conversion")
            text_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
            
            if text_cols:
                manual_col = st.selectbox("Select column to convert:", ["None"] + text_cols)
                
                if manual_col != "None":
                    st.write(f"Sample values from {manual_col}:")
                    st.write(processed_df[manual_col].head(5).tolist())
                    
                    if st.button(f"Convert {manual_col} to numeric"):
                        processed_df[f"{manual_col}_numeric"] = processed_df[manual_col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        processed_df[f"{manual_col}_numeric"] = pd.to_numeric(processed_df[f"{manual_col}_numeric"], errors='coerce')
                        st.session_state.processed_df = processed_df
                        st.success(f"Created {manual_col}_numeric")
                        st.rerun()
        
        with preprocess_tabs[2]:  # Data Type Conversion
            st.subheader("Data Type Conversion")
            
            for col in processed_df.columns:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"{col}")
                    st.write(f"Current: {processed_df[col].dtype}")
                
                with col2:
                    new_type = st.selectbox(
                        "Convert to:",
                        ["Keep as is", "int", "float", "string", "datetime", "category"],
                        key=f"type_{col}"
                    )
                
                with col3:
                    if st.button(f"Convert", key=f"convert_{col}"):
                        try:
                            if new_type == "int":
                                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('Int64')
                            elif new_type == "float":
                                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                            elif new_type == "string":
                                processed_df[col] = processed_df[col].astype(str)
                            elif new_type == "datetime":
                                # Handle date conversion properly
                                if processed_df[col].dtype == 'object':
                                    # Try multiple date formats
                                    processed_df[col] = pd.to_datetime(
                                        processed_df[col], 
                                        errors='coerce',
                                        infer_datetime_format=True
                                    )
                                else:
                                    st.warning("Can only convert object/string columns to datetime")
                            elif new_type == "category":
                                processed_df[col] = processed_df[col].astype('category')
                            
                            st.session_state.processed_df = processed_df
                            st.success(f"Converted {col} to {new_type}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error converting {col}: {str(e)}")
        
        with preprocess_tabs[3]:  # Text Processing
            st.subheader("Text Processing")
            
            text_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
            
            if text_cols:
                selected_text_col = st.selectbox("Select text column to process:", text_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Available Operations:")
                    operations = st.multiselect(
                        "Select operations:",
                        ["Lowercase", "Uppercase", "Remove extra spaces", "Remove special characters", "Extract numbers"]
                    )
                
                with col2:
                    if st.button("Apply Text Operations"):
                        new_col_name = f"{selected_text_col}_processed"
                        processed_df[new_col_name] = processed_df[selected_text_col].astype(str)
                        
                        for op in operations:
                            if op == "Lowercase":
                                processed_df[new_col_name] = processed_df[new_col_name].str.lower()
                            elif op == "Uppercase":
                                processed_df[new_col_name] = processed_df[new_col_name].str.upper()
                            elif op == "Remove extra spaces":
                                processed_df[new_col_name] = processed_df[new_col_name].str.strip().str.replace(r'\s+', ' ', regex=True)
                            elif op == "Remove special characters":
                                processed_df[new_col_name] = processed_df[new_col_name].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                            elif op == "Extract numbers":
                                processed_df[f"{selected_text_col}_numbers"] = processed_df[selected_text_col].str.extract(r'(\d+)')
                        
                        st.session_state.processed_df = processed_df
                        st.success(f"Text processing complete! Created {new_col_name}")
                        st.rerun()
        
        with preprocess_tabs[4]:  # Feature Engineering
            st.subheader("Feature Engineering")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Create New Features:")
                
                # Date features
                date_cols = processed_df.select_dtypes(include=['datetime64']).columns.tolist()
                if date_cols:
                    st.write("Extract Date Components:")
                    selected_date_col = st.selectbox("Select date column:", date_cols)
                    
                    if st.button("Extract Date Features"):
                        processed_df[f"{selected_date_col}_year"] = processed_df[selected_date_col].dt.year
                        processed_df[f"{selected_date_col}_month"] = processed_df[selected_date_col].dt.month
                        processed_df[f"{selected_date_col}_day"] = processed_df[selected_date_col].dt.day
                        processed_df[f"{selected_date_col}_weekday"] = processed_df[selected_date_col].dt.day_name()
                        st.session_state.processed_df = processed_df
                        st.success("Date features extracted!")
                        st.rerun()
                
                # Binning
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.write("Create Bins:")
                    bin_col = st.selectbox("Select numeric column to bin:", numeric_cols)
                    n_bins = st.slider("Number of bins:", 2, 10, 5)
                    
                    if st.button("Create Bins"):
                        processed_df[f"{bin_col}_binned"] = pd.cut(processed_df[bin_col], bins=n_bins, labels=False)
                        st.session_state.processed_df = processed_df
                        st.success(f"Created bins for {bin_col}!")
                        st.rerun()
            
            with col2:
                st.write("Mathematical Operations:")
                
                if len(numeric_cols) >= 2:
                    col_a = st.selectbox("Select first column:", numeric_cols, key="math_col_a")
                    operation = st.selectbox("Operation:", ["+", "-", "*", "/"])
                    col_b = st.selectbox("Select second column:", numeric_cols, key="math_col_b")
                    
                    new_col_name = st.text_input("New column name:", f"{col_a}{operation}{col_b}")
                    
                    if st.button("Create Feature"):
                        try:
                            if operation == "+":
                                processed_df[new_col_name] = processed_df[col_a] + processed_df[col_b]
                            elif operation == "-":
                                processed_df[new_col_name] = processed_df[col_a] - processed_df[col_b]
                            elif operation == "*":
                                processed_df[new_col_name] = processed_df[col_a] * processed_df[col_b]
                            elif operation == "/":
                                processed_df[new_col_name] = processed_df[col_a] / processed_df[col_b]
                            
                            st.session_state.processed_df = processed_df
                            st.success(f"Created feature: {new_col_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating feature: {str(e)}")
        
        # Final processed data preview
        st.markdown("---")
        st.header("4. Processed Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            st.metric("Processed Shape", f"{processed_df.shape[0]} × {processed_df.shape[1]}")
        with col3:
            st.metric("Missing Values", processed_df.isnull().sum().sum())
        
        # Convert datetime columns to string for display
        display_df = processed_df.head(10).copy()
        for col in display_df.select_dtypes(include=['datetime64']).columns:
            display_df[col] = display_df[col].astype(str)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Save processed data
        st.markdown("---")
        st.header("5. Save Processed Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save as Session Data", type="primary"):
                st.session_state.main_df = processed_df
                st.success("✅ Data saved to session! You can now use it in other modules.")
        
        with col2:
            # Convert datetime columns to string for CSV export
            export_df = processed_df.copy()
            for col in export_df.select_dtypes(include=['datetime64']).columns:
                export_df[col] = export_df[col].astype(str)
                
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("🔄 Reset to Original"):
                st.session_state.processed_df = df.copy()
                st.success("Reset to original data")
                st.rerun()

# Helper function to integrate with main app
def get_uploaded_data():
    """
    Function to get uploaded/processed data for use in other modules
    """
    if 'main_df' in st.session_state:
        return st.session_state.main_df
    elif 'processed_df' in st.session_state:
        return st.session_state.processed_df
    elif 'uploaded_data' in st.session_state:
        return st.session_state.uploaded_data
    else:
        # Fallback to default dataset
        try:
            return pd.read_csv("marketing_campaign_dataset.csv")
        except:
            return None