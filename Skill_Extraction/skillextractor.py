import pandas as pd
import json
# Load data from CSV
df = pd.read_csv("./student_data.csv")
df = df.dropna(subset=["skills[0]"])
# Extract skills column
df["skills[0]"] = df["skills[0]"].astype(str).fillna("").apply(lambda x: x.split(","))  # Assuming skills are comma-separated


# skill_list=set()
# for x in df["skills[0]"]:
#     for y in x:
#         if y not in skill_list:
#             skill_list.add(y)

# skill_list=sorted(skill_list)
with open("candidate_skills.json", "w") as file:
    json.dump(df[["_id", "skills[0]"]].to_dict(orient="records"), file, indent=4)