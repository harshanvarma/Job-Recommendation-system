import pandas as pd
import numpy as np
import os
import re
import faiss  # FAISS library for similarity search
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from preprocessing import normalize_skills_batch  # Ensure this module is available

def main():
    # User preference categories (replace with actual user inputs)
    categories = ['Data Science & Analytics - Other']  # This should come from user input or function parameters

    # User's query skills (replace with actual user input)
    user_skills = ['java', 'machine learning', 'python', 'sql', 'analytics']  # This should come from user input
    user_skills = normalize_skills_batch(user_skills)

    # Function to create a safe filename from the category name
    def get_category_filename(category):
        safe_category = re.sub(r'[<>:"/\\|?*]', '_', category)
        filename = f"{safe_category}_subset.csv"
        filepath = os.path.join('subsets', filename)
        return filepath

    # List to hold DataFrames from each category
    dataframes = []

    # Check for each category's CSV file and load it
    for category in categories:
        filepath = get_category_filename(category)
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath, header=None)
            dataframes.append(df)
            print(f"Loaded data for category '{category}' from '{filepath}'")
        else:
            print(f"Warning: Category '{category}' not found in subsets. Skipping this category.")

    # If there is data to process, combine it into one DataFrame
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
    else:
        print("No data to process. Exiting.")
        return

    # Remove rows with NaN values in column 0 (assuming this column contains job postings)
    combined_df = combined_df.dropna(subset=[0])

    # Fill NaN values in column 10 with empty strings (assuming this column contains skills)
    combined_df[10] = combined_df[10].fillna('')

    # Convert skills in column 10 from comma-separated strings to lists and strip whitespace
    combined_df[10] = combined_df[10].apply(lambda x: [skill.strip().lower() for skill in x.split(',')] if isinstance(x, str) else x)

    # Normalize skills in each row
    combined_df[10] = combined_df[10].apply(lambda skills: normalize_skills_batch(skills))

    # Flatten all skills into a single list to create the binarizer's classes
    all_skills = [skill for sublist in combined_df[10] for skill in sublist]

    # **Combine unique skills with user's skills**
    unique_skills = sorted(list(set(all_skills + user_skills)))  # Include user skills

    # Initialize MultiLabelBinarizer with the unique skills
    mlb = MultiLabelBinarizer(classes=unique_skills)
    mlb.fit(combined_df[10])

    # Transform the job postings' skills into binary vectors
    skill_vectors = mlb.transform(combined_df[10])

    # **Padding to make num_bits a multiple of 8**
    num_bits_original = skill_vectors.shape[1]
    num_bits_padded = ((num_bits_original + 7) // 8) * 8  # Next multiple of 8
    padding_required = num_bits_padded - num_bits_original

    if padding_required > 0:
        # Pad skill_vectors with zeros
        skill_vectors = np.hstack([skill_vectors, np.zeros((skill_vectors.shape[0], padding_required), dtype=np.uint8)])

    num_bits = skill_vectors.shape[1]  # Updated number of bits after padding

    # Convert skill_vectors to bytes for FAISS binary index
    skill_vectors_bin = np.packbits(skill_vectors, axis=1)

    # Initialize FAISS binary HNSW index for binary vectors
    index = faiss.IndexBinaryHNSW(num_bits, 32)

    # Optionally set efConstruction and efSearch parameters
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16

    # Add the binary skill vectors to the FAISS index
    index.add(skill_vectors_bin)

    # Transform the user's skills into a binary vector using the same binarizer
    query_skill_vector = mlb.transform([user_skills])

    # **Pad the query vector to match num_bits**
    if padding_required > 0:
        query_skill_vector = np.hstack([query_skill_vector, np.zeros((query_skill_vector.shape[0], padding_required), dtype=np.uint8)])

    # Check if the query vector is all zeros
    if query_skill_vector.sum() == 0:
        print("Warning: None of the user's skills match the skills in the dataset.")
        return

    # Convert the query vector to bytes
    query_skill_vector_bin = np.packbits(query_skill_vector, axis=1)

    # Perform search to get top k recommendations
    k = 5  # Number of recommendations
    distances, indices = index.search(query_skill_vector_bin, k)

    # Retrieve recommended job postings based on indices
    recommended_jobs = combined_df.iloc[indices[0]].reset_index(drop=True)

    # Display the recommended job postings (assuming column 0 contains the job title or description)
    print("\nRecommended Job Postings:")
    for i in range(len(recommended_jobs)):
        job = recommended_jobs.iloc[i]
        distance = distances[0][i]
        print(f"- {job[0]} (Hamming Distance: {distance})")  # Distance is the Hamming distance

    print('\nRecommendation process completed.')

if __name__ == "__main__":
    main()