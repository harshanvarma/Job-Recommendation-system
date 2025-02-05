import pandas as pd
import json
from rapidfuzz import process, fuzz
from fuzzywuzzy import process, fuzz
import re
from nltk.stem import PorterStemmer

# Skill mapping dictionary


def read_json_file(filename):
    # Load the JSON file into a pandas DataFrame
    with open(filename, 'r') as file:
        return json.load(file)

stemmer = PorterStemmer()

# Stop words to be removed


def normalize_skills_batch(skills):
    skill_mapping = {
        "redux":"Redux",
        "ux":"UI and UX",
        "fastapi":"FastAPI",
        # Programming Languages
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "pyspark": "PySpark",
        "python":"python",
        "javascript":"javascript",
        "java": "java",
        "c++": "c++",
        "c#": "c#",
        "php": "php",
        "ruby": "ruby",
        "go": "go",
        "swift": "swift",
        "kotlin": "kotlin",
        "r": "r",
        "scala": "scala",
        "perl": "perl",
        "rust": "rust",
        "dart": "dart",

        # Web Development
        "frontend":"Front end development",
        "frontenddeveloper":"Front end development",
        "frontenddevelopment":"Front end development",
        "Front end development":"Front end development",
        "fullstack":"full stack web development",
        "fullstackdevelopment":"full stack web development",
        "webtechnologies": "full stack web development",
        "webdevelopment": "full stack web development",
        "mern":"Mern stack development",
        "mernstack":"Mern stack development",
        "javafullstack":"Java Full Stack",
        "pythonfullstack":"Python full stack",
        "fullstackjava":"Java Full Stack",
        "fullstackpython":"Python full stack",

        "html": "html",
        "css": "css",
        "sass": "sass",
        "less": "less",
        "react": "react.js",
        "react.js":"react.js",
        "next":"next.js",
        "nextjs":"next.js",
        "angular": "angular",
        "vue": "vue.js",
        "vue.js":"vue.js",
        "node": "node.js",
        "node.js":"node.js",
        "express": "express.js",
        "express.js":"express.js",
        "django": "django",
        "flask": "flask",
        "laravel": "laravel",
        "spring": "spring framework",
        "sprintboot":"spring framework",
        "asp.net": "asp.net",
        "blockchain":"Block chain",
        "datawarehousing":"Data Warehousing",
        "snowflake":"snowflake",
        "sitereliabilityengineering":"Site Reliability Engineering",
        "sre":"Site Reliability Engineering",
        "agile": "Agile",
        "testcases": "Test cases",
        # Databases
        "sql": "sql",
        "sqlqueries":"SQL",
        "structuredquerylanguage":"sql",
        "nosql": "nosql",
        "mysql": "mysql",
        "postgresql": "postgresql",
        "mongodb": "mongodb",
        "redis": "redis",
        "cassandra": "cassandra",
        "oracle": "oracle database",
        "sqlite": "sqlite",
        "firebase": "firebase",
        "dynamodb": "amazon dynamodb",

        # Cloud & DevOps
        "cloudsecurity":"Cloud Security",
        "cloudprotection":"Cloud Security",
        "aws": "amazon web services",
        "amazon web services":"amazon web services",
        "gcp": "google cloud platform",
        "google cloud platform":"google cloud platform",
        "azure": "microsoft azure",
        "microsoft azure":"microsoft azure",
        "docker": "docker",
        "kubernetes": "kubernetes",
        "terraform": "terraform",
        "ansible": "ansible",
        "jenkins": "jenkins",
        "ci/cd": "continuous integration/deployment",
        "continuous integration/deployment":"continuous integration/deployment",
        "git": "git",
        "github": "github",
        "gitlab": "gitlab",
        "bitbucket": "bitbucket",
        "vs":"Visual Studio Code",
        "vscode":"Visual Studio Code",
        

        # Data Science & Machine Learning
        "analytics":"Analytics",
        "bimanager": "BI Manager",
        "businessintelligencemanager":"BI Manager",
        "businessintelligencearchitect":"BI Architect",
        "biarchitect":"BI Architect",
        "bi":"Business intelligence developer",
        "businessintelligence":"Business intelligence developer",
        "databricks":"Data Bricks",
        "googleanlaytics":"Data analysis",
        "dataanalysis":"Data analysis",
        "visualanalytics":"Data analysis",
        "visualizingdata":"Data visualization",
        "datavisualization":"Data visualization",
        "datavalidation":"Data validation",
        "ml": "machine learning",
        "machine learning":"machine learning",
        "dl": "deep learning",
        "deep learning":"deep learning",
        "ai": "artificial intelligence",
        "artificial intelligence":"artificial intelligence",
        "nlp": "natural language processing",
        "processautomation": "Process automation",
        "naturallanguageprocessing":"natural language processing",
        "cv": "computer vision",
        "opencv":"computer vision",
        "computervision": "computer vision",
        "tensorflow": "tensorflow",
        "pytorch": "pytorch",
        "keras": "keras",
        "pandas": "pandas",
        "numpy": "numpy",
        "pycharm":"pycharm integraded environment",
        "jupyter":"jupyter notebook IDE",
        "excel":"MicroSoft Excel",
        "scipy": "scipy",
        "jira": "JIRA",
        "projectdelivery": "Project delivery",
        "projectmanagement": "Project management",
        "billing": "Billing",
    "financialservices": "Financial services",
        "timeseriesanalysis": "Time series analysis",
        "crm": "CRM (Customer Relationship Management)",
    "analyticalskills": "Analytics",
    "translation": "Translation",
        "scikitlearn": "scikitlearn",
        "apachekafka":"Apache Kafka",
        "kafka":"Apache Kafka",
        "spark": "apache spark",
        "apache spark":"apache spark",
        "hadoop": "apache hadoop",
        "apache hadoop":"apache hadoop",
        "tableau": "tableau",
        "powerbi": "power bi",
        "powerpoint":"MS PowerPoint Presentation",
        "mspowerpoint":"MS PowerPoint Presentation",
        "msppt":"MS PowerPoint Presentation",
        "seaborn":"seaborn",
        "matplotlib":"matplotlib",

        # Networking & Security
        "vpn": "virtual private network",
        "dns": "domain name system",
        "http": "hypertext transfer protocol",
        "https": "http secure",
        "ssl": "secure sockets layer",
        "tls": "transport layer security",
        "ssh": "secure shell",
        "ftp": "file transfer protocol",
        "tcp/ip": "transmission control protocol/internet protocol",
        "firewall": "firewall",
        "pen testing": "penetration testing",
        "testing":"testing",
        "alteryx":"Alteryx",
        "communicationskills":"Communication Skills",
        "problemsolving":"Problem Solving",
        # Other Common Abbreviations
        "api": "application programming interface",
        "rest": "representational state transfer",
        "graphql": "graphql",
        "json": "javascript object notation",
        "xml": "extensible markup language",
        "yaml": "yaml ain't markup language",
        "cli": "command line interface",
        "gui": "UI and UX",
        "ui":"UI and UX",
        "uiux":"UI and UX",
        "ide": "integrated development environment",
        "sdk": "Software development",
        "softwaredevelopment":"Software development",
        "softwareengineering":"Software development",
        "oop": "object-oriented programming",
        "fp": "functional programming",
        "tdd": "test-driven development",
        "bdd": "behavior-driven development",

        #marketing
        "marketing":"Marketing",
        "sales":"Sales",
        "ethicalhacking": "Ethical Hacking",
        "cybersecurity": "Cybersecurity",
        "operations":"Operations",
        "ops":"Operations",
        
        # Databases
        "mysql": "MySQL",
        "postgresql": "PostgreSQL",
        "mongodb": "MongoDB",
        "redis": "Redis",
        "oracledatabase": "Oracle Database",
        "sqlite": "SQLite",
        "dynamodb": "DynamoDB",
        
        # Special Cases (Your Examples)
        "c": "c",
        "cplusplus": "c++", 
        "csarp": "c#",
        "dotnet": ".NET",
        "springboot": "Spring Boot",
        "graphql": "GraphQL",
        "restapi": "REST API",
        "oop": "OOP",
        "fpmodeling": "FP Modeling",
        "microsoftoffice":"Microsoft Office",
        "msoffice":"Microsoft Office",

        "googlecolab": "Google Colab",
        "rprogramming": "R Programming",
        "dsa": "Data Structures & Algorithms",
        "datastructures": "Data structures & Algorithms",
        "devops": "DevOps",
        "bigdata": "Big Data",
        "etl": "ETL",
        "digitalmarketing": "Digital Marketing",
        "seo": "SEO",
        "sap": "SAP",
        "erp": "ERP",
        "recruitment": "Recruitment",
        "simon": "SIMON",
    "saas": "SaaS",
    "financialreporting": "Financial reporting",
    "industrialproducts": "Industrial products",
    "accounting": "Accounting",
    "forecasting": "Forecasting",
    "teamwork": "Teamwork",
    "technicalsupport": "Technical support",
    "medicine": "Medicine",
    "userstories":"User stories",

    }
    normalized_skills = []
    
    # Stop words to be removed
    stop_words = {"a", "and", "the", "in", "on", "of", "for", "to", "with", "as", "by", "an", "be", "it", "at", 
                  "basics", "developer", "programmer","programming", "development", "engineer", "using", "skills"}

    for skill in skills:
        # Step 1: Lowercase and strip extra spaces
        skill = skill.lower().strip()

        # Step 2: Replace special characters (&, /, -, ., :, ,) with spaces
        skill = re.sub(r"[&/,\-:.\s]+", " ", skill).strip()

        # Step 3: Remove spaces between words and check for fuzzy match
        temp = skill.split()
        skill = [x for x in temp if x not in stop_words]
        skill = " ".join(skill)
        no_space_skill = skill.replace(" ", "")  # Remove all spaces to create a single string

        # Perform fuzzy matching on the whole string without spaces
        best_match, score = process.extractOne(no_space_skill, skill_mapping.keys(), scorer=fuzz.ratio)[:2]

        if score > 80:  # Threshold for good match
            normalized_skills.append(skill_mapping[best_match])  # Add normalized match
        else:
            # Step 4: Tokenize the skill and remove stop words
            tokens = [token for token in skill.split()]

            # Step 5: Normalize each token using fuzzy matching
            normalized_tokens = []
            for token in tokens:
                # Use fuzzy matching to find the best match for each token
                best_match, score = process.extractOne(token, skill_mapping.keys(), scorer=fuzz.ratio)[:2]

                if score > 80:  # Threshold for fuzzy match
                    normalized_tokens.append(skill_mapping[best_match])  # Map to normalized skill
   # Keep original token if no match

            # Step 6: Add the normalized tokens to the final list, ensuring uniqueness
            normalized_skills.extend(normalized_tokens)

    # Remove duplicates while preserving order
    normalized_skills = list(dict.fromkeys(normalized_skills))

    return normalized_skills
# # Load JSON data
# data = read_json_file("extracted_skills.json")


# df = pd.DataFrame(data, columns=["skills"]) 

# normalized_skills= normalize_skills_batch(df['skills'], skill_mapping)
# dfnorm=pd.DataFrame()
# dfnorm['skills']=normalized_skills



# skills_dict = {"skills": dfnorm['skills'].explode().dropna().unique().tolist()}

# with open("new_skills.json", "w") as f:
#     json.dump(skills_dict, f, indent=4)

# print("Normalized skills saved to skills.json")

# skills_to_normalize = [
#     "mern stack"
# ]

# normalized_skills = normalize_skills_batch(skills_to_normalize, skill_mapping)

# # Print the results
# print(normalized_skills)