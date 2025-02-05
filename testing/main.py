import pandas as pd
import networkx as nx
from preprocessing import normalize_skills_batch
import faiss
from sklearn.preprocessing import MultiLabelBinarizer

# Load and preprocess data
file_path = 'analytics_jobs.csv'
job_postings_data = pd.read_csv(file_path, encoding='iso-8859-1')
job_postings_data = job_postings_data.dropna(subset=['Skills'])
job_postings_data['Skills'] = job_postings_data['Skills'].str.split(',')

# Create unique skills set
unique_skills = set(skill.strip().lower().replace(' ','') for sublist in job_postings_data['Skills'] for skill in sublist)
unique_skills = normalize_skills_batch(unique_skills)
print('total no of columns',len(unique_skills))

# Vectorize skills
mlb = MultiLabelBinarizer()
skill_vectors = mlb.fit_transform(job_postings_data['Skills'])

# Create FAISS index
d = skill_vectors.shape[1]
M = 32
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 64
index.hnsw.efSearch = 32
index.add(skill_vectors.astype('float32'))

# Create a directed graph
G = nx.DiGraph()

# Add job nodes and their required skills
for idx, row in job_postings_data.iterrows():
    job_title = row['Title']
    G.add_node(job_title, type='job')
    for skill in row['Skills']:
        normalized_skills = normalize_skills_batch([skill.strip().lower()])
        if normalized_skills:
            normalized_skill = normalized_skills[0]
            G.add_node(normalized_skill, type='skill')
            G.add_edge(normalized_skill, job_title, relation='required')

# Function to get job recommendations for a skill
def get_job_recommendations(skill, k=5):
    query_skill_vector = mlb.transform([[skill]]).astype('float32')
    D, I = index.search(query_skill_vector, k)
    return job_postings_data.iloc[I[0]]['Title'].tolist()

# # Add edges from skills to recommended jobs
for skill in unique_skills:
    recommended_jobs = get_job_recommendations(skill)
    for job in recommended_jobs:
        G.add_edge(skill, job, relation='recommended')

# Create a list to store all graph data
graph_data = []

# Add node data
for node, attributes in G.nodes(data=True):
    graph_data.append({
        'source': node,
        'target': '',
        'type': attributes['type'],
        'relation': ''
    })

# Add edge data
for source, target, attributes in G.edges(data=True):
    graph_data.append({
        'source': source,
        'target': target,
        'type': '',
        'relation': attributes['relation']
    })

# Convert to DataFrame
graph_df = pd.DataFrame(graph_data)

# Save to CSV
csv_path = 'graph_data.csv'
graph_df.to_csv(csv_path, index=False)

print(f"Graph data saved to {csv_path}")
