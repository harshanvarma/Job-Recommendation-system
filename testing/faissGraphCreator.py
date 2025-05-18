import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from preprocessing2 import preprocess

# Directory containing CSV files
CSV_DIR = './subsets'
INDEX_DIR = './indexes'
os.makedirs(INDEX_DIR, exist_ok=True)

# Load the sentence transformer model
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS HNSW index parameters
DIMENSION = 384  # Embedding size for 'all-MiniLM-L6-v2'
M = 32           # Number of neighbors in HNSW graph
EF_CONSTRUCTION = 200  # Search depth for HNSW graph

def process_csv(csv_file):
    """Process a single CSV file and create FAISS index."""
    category_name = csv_file.replace('_subset.csv', '')
    print(f"\n📌 Processing category: {category_name}")

    csv_path = os.path.join(CSV_DIR, csv_file)
    
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")
        return

    print("🔍 CSV Columns:", df.columns)
    
    if 10 not in df.columns:
        print(f"❌ Column 10 not found in {csv_file}. Available columns: {df.columns}")
        return
    
    # Preprocess text data
    print("⚙️ Preprocessing text data...")
    import re
    import unicodedata

    def preprocess_text(text):
        # Ensure text is a string
        if not isinstance(text, str):
            return ''
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        # Lowercase
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    df[10] = df[10].astype(str).apply(preprocess_text)
    
    # Remove rows with empty text after preprocessing
    df = df[df[10].str.strip() != '']
 
    
    # Convert the text data to a list
    texts = df[10].tolist()
    
    if not texts:
        print("❌ Preprocessing returned an empty list. Skipping file.")
        return

    # Convert text to embeddings
    print("🧠 Generating embeddings...")
    embeddings = MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    if embeddings is None or embeddings.shape[0] == 0:
        print("❌ No embeddings generated. Skipping file.")
        return

    print(f"✅ Embedding shape: {embeddings.shape}")

    # Ensure embeddings are 2D
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    print("⚡ Building FAISS HNSW index...")
    index = faiss.IndexHNSWFlat(DIMENSION, M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(embeddings)

    # Save the FAISS index
    index_file = os.path.join(INDEX_DIR, f"{category_name}_hnsw.index")
    faiss.write_index(index, index_file)
    print(f"✅ Index saved: {index_file}")

    # Save the index-to-data mapping
    mapping_file = os.path.join(INDEX_DIR, f"{category_name}_mapping.npy")
    np.save(mapping_file, df.index.to_numpy())
    print(f"✅ Mapping saved: {mapping_file}")

def main():
    """Main function to process all CSV files."""
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('_subset.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in the directory.")
        return

    for csv_file in csv_files:
        process_csv(csv_file)

if __name__ == "__main__":
    main()
