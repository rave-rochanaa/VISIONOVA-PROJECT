import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import time

# Configuration
TEXT_COLS = ['Campaign_Type', 'Target_Audience', 'Channel_Used', 'Location', 'Language']
NUMERIC_COLS = ['Conversion_Rate', 'Acquisition_Cost', 'ROI', 'Clicks', 'Impressions', 'Engagement_Score']
DATE_COL = 'Date'

def preprocess_data(df):
    """Clean and prepare data with validation and proper type handling"""
    required_columns = TEXT_COLS + NUMERIC_COLS + [DATE_COL, 'Campaign_ID']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Date handling
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')  # Handle errors in date conversion
    df['Date_ts'] = df[DATE_COL].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)

    # Text preprocessing
    df['Combined_Text'] = df[TEXT_COLS].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Numeric preprocessing: clean and convert
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    numeric_scaler = StandardScaler()
    df[NUMERIC_COLS] = numeric_scaler.fit_transform(df[NUMERIC_COLS])

    return df, numeric_scaler

def vectorize_text(corpus):
    """Create optimized text vectorization pipeline"""
    # Fit TF-IDF first to get feature count
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)

    n_features = tfidf_matrix.shape[1]
    n_components = min(100, n_features)  # Ensure n_components <= n_features

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    # Pipeline for transforming queries later
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('svd', svd)
    ])

    return svd_matrix, pipeline

def extract_query_features(query, df, vectorizer, scaler):
    """Extract and process both text and numeric features from query"""
    # Text features
    text_vector = vectorizer.transform([query])

    # Numeric features
    numeric_features = {}
    for col in NUMERIC_COLS:
        # Find exact matches (e.g., "ROI: 5.2")
        exact_match = re.search(fr'\b{col}[\s:=]+(\d+\.?\d*)\b', query, re.IGNORECASE)
        if exact_match:
            numeric_features[col] = float(exact_match.group(1))

        # Handle comparative terms (e.g., "ROI over 5")
        comp_match = re.search(fr'\b{col}\s+(over|above|below|under)\s+(\d+\.?\d*)\b', query, re.IGNORECASE)
        if comp_match:
            numeric_features[col] = float(comp_match.group(2))

    # Create scaled feature array
    default_values = pd.DataFrame(df[NUMERIC_COLS].mean()).T
    default_values.update(pd.DataFrame([numeric_features]))
    scaled_values = scaler.transform(default_values)

    # Convert to dense if needed
    text_vector_dense = text_vector.toarray() if hasattr(text_vector, 'toarray') else text_vector

    return np.concatenate([text_vector_dense, scaled_values], axis=1)

def apply_filters(df, query):
    """Apply sophisticated filters with range detection"""
    filtered = df.copy()

    # Date filtering
    dates = re.findall(r'\b(\d{4}-\d{2}-\d{2})\b', query)
    if dates:
        dates = [pd.to_datetime(d).timestamp() for d in dates]
        filtered = filtered[filtered['Date_ts'].between(min(dates), max(dates))]

    # Location filtering
    locations = [loc for loc in df['Location'].unique() if loc.lower() in query.lower()]
    if locations:
        filtered = filtered[filtered['Location'].isin(locations)]

    # Engagement filtering
    if "high engagement" in query.lower():
        threshold = df['Engagement_Score'].quantile(0.75)
        filtered = filtered[filtered['Engagement_Score'] >= threshold]

    return filtered

def main():
    st.title("AI-Powered Campaign Analytics Engine")

    uploaded_file = st.file_uploader("Upload Campaign Data (CSV)", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis")
        return

    try:
        df = pd.read_csv(uploaded_file)
        df_preprocessed, numeric_scaler = preprocess_data(df)
        text_vectors, text_vectorizer = vectorize_text(df_preprocessed['Combined_Text'])
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return

    query = st.text_input("Search Campaigns (e.g., 'High ROI video campaigns in NY after 2023-01-01')")

    if query:
        start_time = time.time()

        # Apply filters first
        filtered_df = apply_filters(df_preprocessed, query)
        if filtered_df.empty:
            st.warning("No campaigns match your criteria")
            return

        # Feature extraction
        try:
            query_vector = extract_query_features(query, df_preprocessed, text_vectorizer, numeric_scaler)
        except Exception as e:
            st.error(f"Query processing error: {str(e)}")
            return

        # Prepare data matrix for filtered campaigns
        filtered_indices = filtered_df.index.to_list()
        data_matrix = np.concatenate([
            text_vectors[filtered_indices],
            filtered_df.loc[filtered_indices, NUMERIC_COLS].values
        ], axis=1)

        # Similarity calculation
        similarities = cosine_similarity(query_vector, data_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:5]

        # Display results
        st.subheader(f"Top {len(top_indices)} Matching Campaigns ({(time.time()-start_time):.2f}s)")
        for idx in top_indices:
            campaign = filtered_df.iloc[idx]
            with st.expander(f"Campaign {campaign['Campaign_ID']} | Score: {similarities[idx]:.2f}"):
                cols = st.columns(3)
                cols[0].metric("ROI", f"{campaign['ROI']:.2f}")
                cols[1].metric("Engagement", f"{campaign['Engagement_Score']:.2f}")
                cols[2].metric("Cost", f"${campaign['Acquisition_Cost']:.2f}")

                st.caption(f"Channel: {campaign['Channel_Used']} | Location: {campaign['Location']}")
                st.write(f"Dates: {campaign[DATE_COL].strftime('%b %d, %Y') if pd.notnull(campaign[DATE_COL]) else 'N/A'}")
                if campaign['Conversion_Rate'] >= 0 and campaign['Conversion_Rate'] <= 1:
                    st.progress(campaign['Conversion_Rate'])
                else:
                    # Normalize Conversion Rate progress bar to [0,1]
                    normalized = min(max(campaign['Conversion_Rate'], 0), 1)
                    st.progress(normalized)


main()