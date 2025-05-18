import pandas as pd
import numpy as np
import os
import re
import faiss
from sklearn.preprocessing import MultiLabelBinarizer
import unicodedata

# --- Preprocessing Functions ---
def preprocess_text(text):
    """
    Normalize Unicode, convert to lowercase, remove punctuation,
    and remove extra whitespace.
    """
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_and_split(text):
    """
    Preprocess the text and split it into a list of skills.
    Assumes that skills in the text are comma-separated.
    """
    processed_text = preprocess_text(text)
    # Split on comma and filter out empty tokens
    skills = [skill.strip() for skill in processed_text.split(',') if skill.strip()]
    return skills

# --- File Path Helpers ---
def get_category_filename(category):
    safe_category = re.sub(r'[<>:"/\\|?*]', '_', category)
    filename = f"{safe_category}_subset.csv"
    filepath = os.path.join('subsets', filename)
    return filepath

def get_category_filenames(category):
    """Generate file paths for category's dataset, FAISS index, and mapping."""
    safe_category = re.sub(r'[<>:"/\\|?*]', '_', category)
    return (
        os.path.join('./subsets', f"{safe_category}_subset.csv"),
        os.path.join('./test_indexes', f"{safe_category}_hnsw.index"),
        os.path.join('./test_indexes', f"{safe_category}_mapping.npy"),
    )

def load_faiss_index(index_file, mapping_file):
    """Load FAISS index and its mapping file."""
    if not os.path.exists(index_file) or not os.path.exists(mapping_file):
        print(f"⚠️ Missing index or mapping file: {index_file}, {mapping_file}")
        return None, None

    print(f"📥 Loading FAISS index from '{index_file}'...")
    index = faiss.read_index(index_file)
    
    print(f"📥 Loading mapping file from '{mapping_file}'...")
    index_to_dataframe_indices = np.load(mapping_file)
    
    return index, index_to_dataframe_indices

# --- Recommendation Generation ---
def generate_recommendations(df, index, index_to_df_map, user_skills):
    """
    Generate job recommendations using the FAISS index and mapping.
    This function applies the same preprocessing (tokenization) used during index building.
    """
    if df.empty or index is None or index_to_df_map is None:
        return []

    # Process column 10: convert each entry to a list of skills using the same function
    df[10] = df[10].astype(str).apply(preprocess_and_split)

    # Build the vocabulary solely from the dataset (do not merge in user_skills)
    all_skills = [skill for sublist in df[10] for skill in sublist]
    unique_skills = sorted(set(all_skills))
    print(f"✅ Vocabulary size (from dataset): {len(unique_skills)}")

    # Initialize MultiLabelBinarizer with the dataset vocabulary
    mlb = MultiLabelBinarizer(classes=unique_skills)
    mlb.fit(df[10])

    # Process user skills: (assumes user_skills are already individual tokens)
    processed_user_skills = [preprocess_text(skill) for skill in user_skills]
    query_vector = mlb.transform([processed_user_skills]).astype(np.float32)

    if query_vector.shape[1] != index.d:
        print(f"❌ Dimension mismatch: Query({query_vector.shape[1]}) ≠ Index({index.d})")
        return []

    # Perform FAISS search
    k = min(5, index.ntotal)  # Request up to 5 results (or fewer if not available)
    distances, faiss_indices = index.search(query_vector, k)

    recommendations = []
    for i in range(len(faiss_indices[0])):
        job_index = faiss_indices[0][i]
        if job_index < len(index_to_df_map):
            df_index = index_to_df_map[job_index]
            # Assuming column 0 holds the job title/description
            recommendations.append((df.iloc[df_index, 0], distances[0][i]))

    return recommendations

# --- Main Function ---
def main():
    categories = ['Data Science & Analytics - Other']
    user_skills = ['java', 'machine learning', 'python', 'sql', 'analytics']
    # Preprocess user skills (if necessary)
    user_skills = [preprocess_text(skill) for skill in user_skills]

    all_recommendations = []

    for category in categories:
        csv_path, index_path, mapping_path = get_category_filenames(category)

        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: CSV file '{csv_path}' not found. Skipping...")
            continue

        df = pd.read_csv(csv_path, header=None)
        index, index_to_df_map = load_faiss_index(index_path, mapping_path)
        recommendations = generate_recommendations(df, index, index_to_df_map, user_skills)

        if recommendations:
            print(f"\n🔹 **Top Recommendations for {category}:**")
            for job, distance in recommendations:
                print(f"- {job} (Distance: {distance})")
            all_recommendations.extend(recommendations)
        else:
            print(f"🚫 No recommendations found for '{category}'.")

if __name__ == "__main__":
    main()
