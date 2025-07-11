import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
import os
import uuid
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def safe_dataframe_display(df, max_rows=1000):
    """Safely display dataframe by handling Arrow incompatible types"""
    display_df = df.copy()
    
    # Handle problematic column types
    for col in display_df.columns:
        dtype = str(display_df[col].dtype)
        
        # Convert problematic types to string for display
        if 'object' in dtype or 'mixed' in dtype:
            # Check if column contains mixed types
            try:
                # Try to keep as is first
                _ = display_df[col].iloc[0]
            except:
                display_df[col] = display_df[col].astype(str)
        
        # Handle datetime columns
        elif 'datetime' in dtype:
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle very large numbers that might cause issues
        elif display_df[col].dtype in ['int64', 'float64']:
            if display_df[col].abs().max() > 1e15:
                display_df[col] = display_df[col].astype(str)
    
    # Limit rows for performance
    if len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
        st.warning(f"Showing first {max_rows} rows of {len(df)} total rows")
    
    return display_df

def load_file_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == '.json':
            df = pd.read_json(uploaded_file)
        elif file_extension == '.parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def quick_clean_data(df):
    """Quick data cleaning"""
    cleaned_df = df.copy()
    
    # Remove completely empty rows and columns
    cleaned_df = cleaned_df.dropna(how='all')
    cleaned_df = cleaned_df.dropna(axis=1, how='all')
    
    # Clean string columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        if cleaned_df[col].dtype == 'object':
            # Convert to string and clean
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            # Replace 'nan' strings with actual NaN
            cleaned_df[col] = cleaned_df[col].replace(['nan', 'None', 'null', ''], np.nan)
    
    return cleaned_df

def simple_upload_page():
    """Simple and robust upload page"""
    st.title("📁 Data Upload & Analysis")
    st.markdown("---")
    
    # File upload section
    st.header("1. Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )
    
    with col2:
        use_sample = st.button("📊 Use Sample Data", type="secondary")
    
    # Load data
    df = None
    
    if uploaded_file is not None:
        with st.spinner("Loading file..."):
            df = load_file_data(uploaded_file)
            if df is not None:
                st.success(f"✅ File loaded! Shape: {df.shape}")
                st.session_state.raw_data = df
                st.session_state.data_source = uploaded_file.name
    
    elif use_sample:
        try:
            df = pd.read_csv("marketing_campaign_dataset.csv")
            st.success(f"✅ Sample data loaded! Shape: {df.shape}")
            st.session_state.raw_data = df
            st.session_state.data_source = "Sample Data"
        except FileNotFoundError:
            st.error("Sample file not found. Please upload your own data.")
    
    # Use existing data if available
    elif 'raw_data' in st.session_state:
        df = st.session_state.raw_data
        st.info(f"📊 Using: {st.session_state.get('data_source', 'Unknown')}")
    
    if df is not None:
        # Quick stats
        st.markdown("---")
        st.header("2. Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory (MB)", f"{memory_mb:.1f}")
        
        # Data preview with safe display
        st.subheader("Data Preview")
        try:
            display_df = safe_dataframe_display(df.head(10))
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Display error: {str(e)}")
            # Fallback to basic info
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Quick preprocessing
        st.markdown("---")
        st.header("3. Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Clean Data"):
                with st.spinner("Cleaning data..."):
                    cleaned_df = quick_clean_data(df)
                    st.session_state.processed_data = cleaned_df
                    st.success("Data cleaned!")
                    st.rerun()
        
        with col2:
            if st.button("💾 Save for Analysis"):
                processed_df = st.session_state.get('processed_data', df)
                st.session_state.main_df = processed_df
                st.success("✅ Data ready for analysis!")
        
        with col3:
            # Download processed data
            processed_df = st.session_state.get('processed_data', df)
            
            # Safe CSV conversion
            try:
                csv_data = processed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Download error: {str(e)}")
        
        # Show processed data if available
        if 'processed_data' in st.session_state:
            st.markdown("---")
            st.subheader("Processed Data Preview")
            try:
                processed_display = safe_dataframe_display(st.session_state.processed_data.head(5))
                st.dataframe(processed_display, use_container_width=True)
            except Exception as e:
                st.write(f"Processed data shape: {st.session_state.processed_data.shape}")

def get_main_data():
    """Get the main dataset for analysis"""
    if 'main_df' in st.session_state:
        return st.session_state.main_df
    elif 'processed_data' in st.session_state:
        return st.session_state.processed_data
    elif 'raw_data' in st.session_state:
        return st.session_state.raw_data
    return None


class RAGProcessor:
    """RAG processor with ChromaDB vector storage and retrieval"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = "all-MiniLM-L6-v2"
        self.documents = []
        self.chunk_size = 500
        self.overlap = 100
        self.collection_name = "marketing_rag"
        self.initialized = False
        
    def initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent client with database path
            self.client = chromadb.PersistentClient(path="chroma_db")
            
            # Create embedding function
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=sentence_transformer_ef,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace
        text = re.sub(r'[^\w\s.,!?@-]', '', text)  # Remove special chars
        return text.strip()
    
    def create_document_chunks(self, df: pd.DataFrame, text_columns: List[str]) -> List[Dict]:
        """Split documents into manageable chunks with metadata"""
        chunks = []
        for idx, row in df.iterrows():
            # Combine text from selected columns
            combined_text = " ".join([
                str(row[col]) if pd.notna(row[col]) else "" 
                for col in text_columns
            ]).strip()
            
            if not combined_text:
                continue
                
            # Split into words
            words = combined_text.split()
            total_words = len(words)
            
            # Create chunks based on word count
            if total_words <= self.chunk_size:
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'text': combined_text,
                    'source_row': idx,
                    'chunk_id': 0,
                    'word_count': total_words,
                    'metadata': {col: row[col] for col in df.columns if col not in text_columns}
                })
            else:
                chunk_id = 0
                for i in range(0, total_words, self.chunk_size - self.overlap):
                    chunk_words = words[i:i + self.chunk_size]
                    chunk_text = " ".join(chunk_words)
                    chunks.append({
                        'id': str(uuid.uuid4()),
                        'text': chunk_text,
                        'source_row': idx,
                        'chunk_id': chunk_id,
                        'word_count': len(chunk_words),
                        'metadata': {col: row[col] for col in df.columns if col not in text_columns}
                    })
                    chunk_id += 1
        return chunks
    
    def build_vector_database(self, chunks: List[Dict]):
        """Create vector database using ChromaDB"""
        if not self.initialize_chroma():
            return 0
        
        # Prepare documents for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk['text'])
            
            # Create metadata with source information
            metadata = chunk['metadata'].copy()
            metadata.update({
                'source_row': chunk['source_row'],
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count']
            })
            metadatas.append(metadata)
            ids.append(chunk['id'])
        
        # Add to ChromaDB collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Store documents locally for reference
        self.documents = chunks
        return len(chunks)
    
    def retrieve_similar_documents(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve most similar documents using ChromaDB"""
        if not self.initialized or self.collection is None:
            return []
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[self.preprocess_text(query)],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            retrieved_docs = []
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                text = results['documents'][0][i]
                
                # Convert distance to similarity score (cosine distance to similarity)
                similarity = 1 - distance
                
                # Create document object
                doc = {
                    'id': doc_id,
                    'text': text,
                    'metadata': metadata
                }
                
                retrieved_docs.append((doc, similarity))
            
            return retrieved_docs
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return []
    
    def generate_context(self, retrieved_docs: List[Tuple[Dict, float]]) -> str:
        """Generate context from retrieved documents"""
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            preview = doc['text'][:500] + ("..." if len(doc['text']) > 500 else "")
            context_parts.append(f"📄 Result {i} (relevance: {score:.3f}):\n{preview}")
        
        return "\n\n".join(context_parts)
    
    def get_document_count(self) -> int:
        """Get number of documents in collection"""
        if self.collection:
            return self.collection.count()
        return 0
    
    def reset_database(self):
        """Reset the vector database"""
        try:
            if self.client:
                self.client.delete_collection(self.collection_name)
                st.success("Vector database reset successfully!")
            self.initialized = False
            self.collection = None
            self.documents = []
            return True
        except Exception as e:
            st.error(f"Error resetting database: {str(e)}")
            return False

def get_uploaded_data():
    """Get data from Streamlit session state"""
    if 'main_df' in st.session_state:
        return st.session_state.main_df
    elif 'processed_df' in st.session_state:
        return st.session_state.processed_df
    elif 'uploaded_data' in st.session_state:
        return st.session_state.uploaded_data
    else:
        try:
            return pd.read_csv("marketing_campaign_dataset.csv")
        except Exception:
            return None

def rag_nlp_page():
    """Main RAG interface with ChromaDB"""
    st.title("🧠 RAG - Document Intelligence Module")
    st.markdown("---")
    st.caption("Powered by ChromaDB vector database and Sentence Transformers embeddings")
    simple_upload_page()

    # Initialize processor
    if 'rag_processor' not in st.session_state:
        st.session_state.rag_processor = RAGProcessor()
        
    rag_processor = st.session_state.rag_processor

    # Get data
    df = get_uploaded_data()
    if df is None:
        st.warning("⚠️ Please upload data in the Upload module first.")
        return

    tabs = st.tabs([
        "📚 Document Processing",
        "🔍 Semantic Search",
        "💬 RAG Chat",
        "📊 Analytics"
    ])

    with tabs[0]:  # Document Processing
        st.header("Document Processing & Chunking")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.info("""
            *How it works:*
            1. Select text columns to include
            2. Configure chunking parameters
            3. Process documents to create vector embeddings
            """)
        
        with col2:
            if rag_processor.initialized:
                doc_count = rag_processor.get_document_count()
                st.metric("Documents in Database", doc_count)
            else:
                st.warning("Database not initialized")

        text_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
        if not text_cols:
            st.error("No text columns found in dataset!")
            return

        selected_text_cols = st.multiselect(
            "Select text columns for RAG processing:",
            text_cols,
            default=text_cols[:min(3, len(text_cols))]
        )

        col1, col2 = st.columns(2)
        with col1:
            rag_processor.chunk_size = st.slider("Chunk size (words):", 50, 1000, 500)
        with col2:
            rag_processor.overlap = st.slider("Chunk overlap (words):", 0, 200, 100)

        if st.button("🔄 Process Documents", type="primary", key="process_docs"):
            if not selected_text_cols:
                st.error("Select at least one text column!")
            else:
                with st.spinner("Processing documents..."):
                    chunks = rag_processor.create_document_chunks(df, selected_text_cols)
                    if not chunks:
                        st.error("No valid chunks created.")
                    else:
                        num_docs = rag_processor.build_vector_database(chunks)
                        if num_docs > 0:
                            st.session_state.rag_config = {
                                'text_columns': selected_text_cols,
                                'chunk_size': rag_processor.chunk_size,
                                'overlap': rag_processor.overlap,
                                'num_documents': num_docs,
                                'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.success(f"✅ Processed {num_docs} document chunks!")
                            
                            # Show stats
                            avg_len = np.mean([chunk['word_count'] for chunk in chunks])
                            st.metric("Average Chunk Length", f"{avg_len:.1f} words")

        st.markdown("---")
        st.subheader("Database Management")
        
        if st.button("🔄 Re-initialize Database", help="Reset and re-create database"):
            if rag_processor.initialize_chroma():
                st.success("Database initialized successfully!")
        
        if st.button("🗑️ Clear Database", type="secondary", help="Delete all documents"):
            if rag_processor.reset_database():
                st.success("Database cleared successfully!")
                if 'rag_config' in st.session_state:
                    del st.session_state.rag_config
                st.rerun()

    with tabs[1]:  # Semantic Search
        st.header("Semantic Document Search")
        
        if not rag_processor.initialized or rag_processor.get_document_count() == 0:
            st.warning("⚠️ Please process documents first")
            return

        query = st.text_area("Enter your query:", height=100, key="rag_query",
                            placeholder="Search for marketing campaigns related to...")
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of results:", 1, 20, 5, key="rag_top_k")
        with col2:
            min_similarity = st.slider("Minimum similarity:", 0.0, 1.0, 0.2, step=0.05)

        if st.button("🔍 Search Documents", key="rag_search"):
            if not query.strip():
                st.warning("Please enter a query")
            else:
                with st.spinner("Searching with semantic AI..."):
                    results = rag_processor.retrieve_similar_documents(query, top_k)
                    # Filter by minimum similarity
                    results = [(doc, score) for doc, score in results if score >= min_similarity]
                    
                    if not results:
                        st.warning("No relevant documents found. Try a different query or lower the similarity threshold.")
                    else:
                        st.success(f"Found {len(results)} relevant documents")
                        
                        # Display similarity distribution
                        scores = [score for _, score in results]
                        fig = px.bar(
                            x=list(range(1, len(scores)+1),
                            y=scores,
                            title="Document Similarity Scores",
                            labels={'x': 'Rank', 'y': 'Similarity'},
                            color=scores,
                            color_continuous_scale='Bluered'
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results
                        for i, (doc, score) in enumerate(results, 1):
                            with st.expander(f"📄 Result {i} - Similarity: {score:.3f}"):
                                st.write(doc['text'])
                                
                                if doc['metadata']:
                                    st.subheader("Metadata:")
                                    st.json({k: v for k, v in doc['metadata'].items() if k not in ['source_row', 'chunk_id', 'word_count']})

    with tabs[2]:  # RAG Chat
        st.header("Conversational RAG Interface")
        
        if not rag_processor.initialized or rag_processor.get_document_count() == 0:
            st.warning("⚠️ Please process documents first")
            return

        # Initialize chat history
        if 'rag_chat_history' not in st.session_state:
            st.session_state.rag_chat_history = []

        # Display chat history
        for i, (user_msg, bot_msg, context) in enumerate(st.session_state.rag_chat_history):
            with st.chat_message("user"):
                st.write(user_msg)
                
            with st.chat_message("assistant"):
                st.write(bot_msg)
                if context:
                    with st.expander(f"View context sources {i+1}"):
                        st.write(context)
            
            st.divider()

        # User input
        user_input = st.chat_input("Ask about your marketing data...")
        
        if user_input:
            # Add user message to history
            st.session_state.rag_chat_history.append((user_input, "", ""))
            
            with st.spinner("Analyzing documents..."):
                # Retrieve relevant documents
                results = rag_processor.retrieve_similar_documents(user_input, 3)
                context = rag_processor.generate_context(results) if results else "No context found"
                
                # Generate response
                if results:
                    response = f"I found {len(results)} relevant documents in our marketing database. "
                    response += f"The most relevant result has a similarity score of {results[0][1]:.3f}.\n\n"
                    response += "Here's a summary of the key information:\n\n"
                    
                    # Create bullet point summary
                    for i, (doc, score) in enumerate(results[:3], 1):
                        summary = doc['text'][:150] + ("..." if len(doc['text']) > 150 else "")
                        response += f"{i}. {summary}\n"
                    
                    response += "\nYou can view the full context in the expander below."
                else:
                    response = "I couldn't find relevant information in our documents. Try rephrasing your question or adding more context."
                
                # Update last message with response
                st.session_state.rag_chat_history[-1] = (user_input, response, context)
                st.rerun()

        if st.button("Clear Chat History", type="secondary"):
            st.session_state.rag_chat_history = []
            st.rerun()

    with tabs[3]:  # Analytics
        st.header("RAG System Analytics")
        
        if not rag_processor.initialized or rag_processor.get_document_count() == 0:
            st.warning("⚠️ Please process documents first")
            return
            
        # Document statistics
        st.subheader("Document Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", rag_processor.get_document_count())
        with col2:
            if rag_processor.documents:
                total_words = sum(doc['word_count'] for doc in rag_processor.documents)
                st.metric("Total Words", f"{total_words:,}")
        with col3:
            if rag_processor.documents:
                avg_len = np.mean([doc['word_count'] for doc in rag_processor.documents])
                st.metric("Avg. Words/Chunk", f"{avg_len:.1f}")
        
        # Visualizations
        if rag_processor.documents:
            col1, col2 = st.columns(2)
            with col1:
                # Document length distribution
                doc_lengths = [doc['word_count'] for doc in rag_processor.documents]
                fig = px.histogram(
                    x=doc_lengths,
                    nbins=20,
                    title="Document Length Distribution",
                    labels={'x': 'Words per Document', 'y': 'Count'},
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Query analytics
        if 'rag_chat_history' in st.session_state and st.session_state.rag_chat_history:
            st.subheader("Query Analytics")
            queries = [msg[0] for msg in st.session_state.rag_chat_history]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Recent Queries:")
                for i, query in enumerate(queries[-5:], 1):
                    st.write(f"{i}. {query[:80]}{'...' if len(query) > 80 else ''}")
            
            with col2:
                # Query length distribution
                query_lengths = [len(q.split()) for q in queries]
                if query_lengths:
                    fig = px.histogram(
                        x=query_lengths,
                        title="Query Length Distribution",
                        labels={'x': 'Words per Query', 'y': 'Frequency'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("System Information")
        st.write(f"Embedding Model: {rag_processor.embedding_model}")
        st.write(f"Database Location: chroma_db/{rag_processor.collection_name}")
        st.write(f"Chunk Size: {rag_processor.chunk_size} words | Overlap: {rag_processor.overlap} words")
        
        if 'rag_config' in st.session_state:
            st.json(st.session_state.rag_config)

# Run the app
if __name__ == "_main_":
    rag_nlp_page()