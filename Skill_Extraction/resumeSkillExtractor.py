import fitz  # PyMuPDF
import json
import spacy
from spacy.matcher import PhraseMatcher

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF resume."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


resume_text = extract_text_from_pdf("./Resumes/1729256225501-Madhuri Gajanan Gadekar.pdf")
print(resume_text)


with open("extracted_skills.json", "r") as file:
    skill_list = json.load(file)  # Example: ["Python", "Machine Learning", "SEO", "Social Media Marketing"]


nlp = spacy.load("en_core_web_sm")  
matcher = PhraseMatcher(nlp.vocab)


patterns = [nlp(skill.lower()) for skill in skill_list]
matcher.add("SKILLS", patterns)

def extract_skills_from_text(text):
    """Extract skills from resume text using PhraseMatcher."""
    extracted_skills = set()
    doc = nlp(text.lower())

    matches = matcher(doc)  # Find skill matches
    for match_id, start, end in matches:
        extracted_skills.add(doc[start:end].text)

    return list(extracted_skills)

skills = extract_skills_from_text(resume_text)
print("Extracted Skills:", skills)
