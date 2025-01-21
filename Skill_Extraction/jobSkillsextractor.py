import json
import spacy
from spacy.matcher import PhraseMatcher
import re

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load predefined skills from extracted_skills.json
with open("extracted_skills.json", "r", encoding="utf-8") as file:
    skill_data = json.load(file)

# Extract the list of skills from the JSON file
skill_list = skill_data.get("skills", [])  # Assuming JSON structure is {"skills": ["Python", "Java", "AWS"]}

# Create PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(skill.lower()) for skill in skill_list]
matcher.add("SKILLS", patterns)

def extract_skills(text):
    """Extract predefined skills from text using PhraseMatcher"""
    if not text:
        return []
    
    doc = nlp(text.lower())  # Convert text to lowercase
    matches = matcher(doc)
    
    # Extract matched skills
    extracted_skills = set()
    for match_id, start, end in matches:
        extracted_skills.add(doc[start:end].text)

    return list(extracted_skills)

def parse_job_postings(file_path):
    """Parse the plain text job postings file into a structured format"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    job_blocks = content.split("--------------------------------------------------")
    job_data = []

    for job in job_blocks:
        job = job.strip()
        if not job:
            continue  # Skip empty blocks

        job_info = {}
        
        # Extract fields using regex
        job_info["Title"] = re.search(r"Title:\s*(.*)", job).group(1) if re.search(r"Title:\s*(.*)", job) else "N/A"
        job_info["Company"] = re.search(r"Company:\s*(.*)", job).group(1) if re.search(r"Company:\s*(.*)", job) else "N/A"
        job_info["Experience"] = re.search(r"Experience:\s*(.*)", job).group(1) if re.search(r"Experience:\s*(.*)", job) else "N/A"
        job_info["Skills"] = re.search(r"Skills:\s*(.*)", job).group(1) if re.search(r"Skills:\s*(.*)", job) else "N/A"
        
        job_data.append(job_info)

    return job_data

def process_job_postings(file_path):
    """Extract and save skills from parsed job postings"""
    job_postings = parse_job_postings(file_path)
    extracted_data = []

    for job in job_postings:
        title = job["Title"]
        company = job["Company"]
        experience = job["Experience"]
        skills = job["Skills"]

        extracted_skills = []

        # If skills are available, use them
        if skills and skills != "N/A":
            extracted_skills = extract_skills(skills)

        extracted_data.append({
            "Title": title,
            "Company": company,
            "Experience": experience,
            "Extracted Skills": extracted_skills
        })

    # Save extracted skills back to JSON
    with open("updated_extracted_skills.json", "w", encoding="utf-8") as file:
        json.dump(extracted_data, file, indent=4)

    print("✅ Extracted skills saved to updated_extracted_skills.json")

# Process the job postings from the plain text file
process_job_postings("./scraped_jobs/foundit.json")
