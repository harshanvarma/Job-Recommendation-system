import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import heapq

# Load candidate and job data
candidates_df = pd.read_json('candidate_skills.json')
jobs_df = pd.read_json('data.json')

# Extract skills and IDs, handling missing values
candidates_skills = candidates_df['skills[0]'].apply(lambda x: x if isinstance(x, list) else []).tolist()
candidates_id = candidates_df['_id'].tolist()
jobs_skills = jobs_df['Extracted Skills'].apply(lambda x: x if isinstance(x, list) else []).tolist()

# Jaccard Similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# **Optimized Cosine Similarity Calculation**
all_skills_text = [' '.join(skills) for skills in (candidates_skills + jobs_skills)]
vectorizer = CountVectorizer().fit(all_skills_text)

candidates_vectors = vectorizer.transform([' '.join(skills) for skills in candidates_skills])
jobs_vectors = vectorizer.transform([' '.join(skills) for skills in jobs_skills])

# **Efficient Recommendation Calculation**
recommendations = []

for idx, (candidate_vector, candidate_skills, candidate_id) in enumerate(zip(candidates_vectors, candidates_skills, candidates_id)):
    candidate_set = set(candidate_skills)
    scores = []
    
    for job_idx, (job_vector, job_skills) in enumerate(zip(jobs_vectors, jobs_skills)):
        job_set = set(job_skills)
        
        # Compute Jaccard similarity
        jaccard_score = jaccard_similarity(candidate_set, job_set)
        
        # Compute Cosine similarity (efficient lookup)
        cosine_score = cosine_similarity(candidate_vector, job_vector)[0][0]
        
        # Compute final score
        average_score = (jaccard_score + cosine_score) / 2
        scores.append((job_idx, average_score))
    
    # Get top 10 jobs efficiently
    top_jobs = heapq.nlargest(10, scores, key=lambda x: x[1])
    job_titles = [jobs_df.iloc[job_idx]['Title'] for job_idx, _ in top_jobs]

    # Store recommendations for CSV export
    recommendations.append({"Candidate ID": candidate_id, "Recommended Jobs": " | ".join(job_titles)})

# **Save recommendations to CSV**
recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv("recommended_jobs.csv", index=False)

print("✅ Recommended jobs saved to 'recommended_jobs.csv'")
