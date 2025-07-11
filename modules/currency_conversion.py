import pandas as pd
import numpy as np
import re
import streamlit as st

def currency_conversion_tab(processed_df):
    """
    Streamlit currency conversion tab that can be imported and called directly
    Handles both automatic detection and manual conversion of currency columns
    Returns: Updated DataFrame after currency conversion
    """
    st.subheader("💰 Currency Conversion")
    st.markdown("Convert columns with currency symbols ($, €, £, etc.) to numeric values for analysis")

    # Helper function for currency conversion
    def convert_currency_column(series, parentheses_negative=False):
        """Convert pandas Series with currency values to numeric floats"""
        cleaned = series.astype(str)
        
        # Handle parentheses as negative values
        if parentheses_negative:
            cleaned = cleaned.str.replace(r'\(([\d,\.]+)\)', r'-\1', regex=True)
        
        # Remove non-numeric characters
        cleaned = cleaned.str.replace(r'[^\d\.-]', '', regex=True)
        cleaned = cleaned.replace(r'^\s*$', np.nan, regex=True)
        
        return pd.to_numeric(cleaned, errors='coerce')

    # Configuration options
    st.markdown("### ⚙️ Conversion Settings")
    col1, col2 = st.columns(2)
    with col1:
        replace_original = st.checkbox("Replace original columns", value=True)
    with col2:
        parentheses_as_negative = st.checkbox("Treat (parentheses) as negative", value=True)

    # Automatic detection section
    st.markdown("### 🔍 Automatic Detection")
    currency_keywords = [
        'price', 'cost', 'amount', 'fee', 'charge', 
        'value', 'revenue', 'expense', 'income', 'payment'
    ]
    
    # Find potential currency columns
    potential_currency_cols = []
    for col in processed_df.columns:
        col_lower = col.lower()
        
        # Check column name hints
        name_match = any(keyword in col_lower for keyword in currency_keywords)
        
        # Check value patterns
        value_match = False
        if processed_df[col].dtype == 'object':
            sample = processed_df[col].astype(str).str.cat(sep=' ')
            value_match = any([
                '$' in sample,
                '€' in sample,
                '£' in sample,
                '¥' in sample,
                '₹' in sample,
                '(' in sample and ')' in sample,
                any(re.search(r'\d{1,3}(,\d{3})+\.?\d*', val)) 
                for val in processed_df[col].dropna().astype(str).head(20)
            ])
        
        if name_match or value_match:
            potential_currency_cols.append(col)
    
    # Automatic conversion UI
    if potential_currency_cols:
        st.success(f"Detected {len(potential_currency_cols)} potential currency columns")
        with st.expander("View detected columns"):
            for col in potential_currency_cols:
                sample_vals = processed_df[col].head(5).tolist()
                st.write(f"- *{col}*: {sample_vals}")
        
        selected_currency_cols = st.multiselect(
            "Select columns to convert:",
            potential_currency_cols,
            default=potential_currency_cols
        )
        
        if st.button("🚀 Convert Selected Columns", key="auto_convert"):
            for col in selected_currency_cols:
                try:
                    converted = convert_currency_column(
                        processed_df[col],
                        parentheses_negative=parentheses_as_negative
                    )
                    
                    if replace_original:
                        processed_df[col] = converted
                        new_col = col
                    else:
                        new_col = f"{col}_numeric"
                        processed_df[new_col] = converted
                        
                    st.success(f"Converted *{col}* → *{new_col}*")
                except Exception as e:
                    st.error(f"Error converting {col}: {str(e)}")
    else:
        st.warning("No currency columns automatically detected")
    
    # Manual conversion section
    st.markdown("---")
    st.subheader("🛠 Manual Conversion")
    all_columns = processed_df.columns.tolist()
    
    if all_columns:
        manual_col = st.selectbox("Select any column to convert:", 
                                 [""] + all_columns)
        
        if manual_col:
            st.write(f"*Sample values from '{manual_col}':*")
            st.write(processed_df[manual_col].head(10).reset_index(drop=True))
            
            if st.button("🔧 Convert Selected Column", key="manual_convert"):
                try:
                    converted = convert_currency_column(
                        processed_df[manual_col],
                        parentheses_negative=parentheses_as_negative
                    )
                    
                    if replace_original:
                        processed_df[manual_col] = converted
                        new_col = manual_col
                    else:
                        new_col = f"{manual_col}_numeric"
                        processed_df[new_col] = converted
                    
                    st.success(f"Converted *{manual_col}* → *{new_col}*")
                    st.write("*Preview:*")
                    st.dataframe(pd.DataFrame({
                        'Original': processed_df[manual_col].head(10),
                        'Converted': converted.head(10)
                    }))
                except Exception as e:
                    st.error(f"Conversion failed: {str(e)}")
    else:
        st.info("No columns available for conversion")
    
    return processed_df