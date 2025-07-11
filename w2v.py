import streamlit as st
import pandas as pd
from word_embedding_utils import csv_to_word2vec, visualize_embeddings, save_embeddings
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Word Embedding Generator", layout="wide")

# App title
st.title("CSV to Word Embeddings Converter")

# Sidebar for parameters
with st.sidebar:
    st.header("Parameters")
    vector_size = st.slider("Embedding Size", 50, 300, 100)
    window = st.slider("Context Window", 2, 15, 5)
    min_count = st.slider("Minimum Word Count", 1, 10, 2)

# Main content area
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Select text column
        text_columns = list(df.columns)
        text_column = st.selectbox("Select text column", text_columns)
        
        if st.button("Generate Embeddings"):
            with st.spinner("Training word embeddings..."):
                # Train Word2Vec model
                model, word_vectors = csv_to_word2vec(
                    csv_file=uploaded_file,
                    text_column=text_column,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count
                )
            
            st.success("Training completed!")
            
            # Show word vectors
            st.subheader("Word Embeddings Visualization")
            fig = visualize_embeddings(word_vectors)
            st.pyplot(fig)
            
            # Show similar words examples
            st.subheader("Example Similar Words")
            cols = st.columns(3)
            for i, word in enumerate(list(model.wv.index_to_key)[:6]):
                with cols[i%3]:
                    if word:
                        similar_words = model.wv.most_similar(word, topn=3)
                        similar_text = "\n".join([f"{word} ({score:.2f})" for word, score in similar_words])
                        st.text_area(f"Similar to '{word}'", value=similar_text, height=100)
            
            # Download embeddings
            st.subheader("Download Embeddings")
            save_embeddings(word_vectors, "word_vectors.txt")
            with open("word_vectors.txt", "rb") as f:
                st.download_button(
                    label="Download Word Vectors",
                    data=f,
                    file_name="word_vectors.txt",
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Instructions
with st.expander("How to use this app"):
    st.markdown("""
    1. Upload a CSV file containing your text data
    2. Select the column containing text documents
    3. Adjust parameters in the sidebar (optional)
    4. Click 'Generate Embeddings'
    5. View results and download word vectors
    
    *Note:* The app will:
    - Automatically preprocess text (lowercase, remove punctuation/numbers)
    - Remove stopwords and short words
    - Train using Word2Vec algorithm
    """)