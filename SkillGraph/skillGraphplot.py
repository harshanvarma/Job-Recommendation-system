import pandas as pd
import networkx as nx

# Load job data
job_data_file = 'data.json'  # Replace with your job JSON file path
jobs_df = pd.read_json(job_data_file)

# Load candidate data
candidate_data_file = 'candidate_skills.json'  # Replace with your candidate JSON file path
candidates_df = pd.read_json(candidate_data_file)

# Create a directed graph
G = nx.DiGraph()

# Add job nodes and their required skills
for index, row in jobs_df.iterrows():
    job_title = row['Title']
    G.add_node(job_title, type='job')
    for skill in row['Extracted Skills']:
        normalized_skill = skill.strip().lower()  # Normalize skill to lowercase
        G.add_node(normalized_skill, type='skill')
        G.add_edge(normalized_skill, job_title)

# Add candidate nodes and their skills
for index, row in candidates_df.iterrows():
    candidate_id = row['_id']
    G.add_node(candidate_id, type='candidate')  # Add candidate node
    for skill in row['skills[0]']:  # Adjust this based on your actual DataFrame structure
        normalized_skill = skill.strip().lower()  # Normalize skill to lowercase
        G.add_node(normalized_skill, type='skill')  # Ensure no leading/trailing spaces
        G.add_edge(candidate_id, normalized_skill)  # Connect candidate to their skills

# Export to GEXF
nx.write_gexf(G, 'job_skills_candidates_graph.gexf')

# Optionally, export to GraphML
