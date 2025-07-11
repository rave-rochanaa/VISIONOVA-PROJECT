import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Preprocess text by removing punctuation, numbers, and stopwords
    Returns empty list for non-string input
    """
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove numbers
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return tokens

def csv_to_word2vec(csv_file, text_column, vector_size=100, window=5, min_count=1, workers=4):
    """
    Process CSV data and train Word2Vec model
    Returns trained model and word vectors
    """
    df = pd.read_csv(csv_file)
    
    if text_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Column '{text_column}' not found. Available columns: {available}")
    
    # Preprocess text data
    corpus = df[text_column].apply(preprocess_text).tolist()
    corpus = [doc for doc in corpus if len(doc) > 0]
    
    if not corpus:
        raise ValueError("No valid text found after preprocessing")
    
    # Train Word2Vec model
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    return model, {word: model.wv[word] for word in model.wv.index_to_key}

def visualize_embeddings(word_vectors, words_to_plot=None, n_components=2):
    """
    Create PCA visualization of word embeddings
    Returns matplotlib figure object
    """
    plt.figure(figsize=(10, 8))
    
    # Select words to visualize
    words_to_plot = words_to_plot or list(word_vectors.keys())[:30]
    valid_words = [word for word in words_to_plot if word in word_vectors]
    
    if not valid_words:
        return plt.gcf()  # Return empty figure
    
    vectors = np.array([word_vectors[word] for word in valid_words])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    points = pca.fit_transform(vectors)
    
    # Create plot
    if n_components == 2:
        plt.scatter(points[:, 0], points[:, 1], c='blue')
        for i, word in enumerate(valid_words):
            plt.annotate(word, (points[i, 0], points[i, 1]))
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        
    elif n_components == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='blue')
        for i, word in enumerate(valid_words):
            ax.text(points[i, 0], points[i, 1], points[i, 2], word)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
    
    plt.title("Word Embeddings Visualization")
    plt.tight_layout()
    return plt.gcf()

def save_embeddings(word_vectors, output_file):
    """
    Save word vectors to text file in Word2Vec format
    """
    with open(output_file, 'w') as f:
        f.write(f"{len(word_vectors)} {len(next(iter(word_vectors.values())))}\n")
        for word, vector in word_vectors.items():
            f.write(f"{word} {' '.join(map(str, vector))}\n")