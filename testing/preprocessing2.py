import spacy
def preprocess(subset_skills,candidate_skills):
    nlp = spacy.load("en_core_web_md")
    hash={}

    for x in subset_skills:
        x=x.lower()
        if x[0] in hash:
            hash[x[0]]+=[x]
        else:
            hash[x[0]]=[x]

    print(hash)

    def find_closest_match(text, skills):
        text_vec = nlp(text)
        skills_vec = list(nlp.pipe(skills))  # Process all skills at once
        best_match = max(skills_vec, key=lambda opt: text_vec.similarity(opt))
        return best_match.text  # Return the matched text

    final=[]
    for x in candidate_skills:
        x=x.lower()
        if(x[0] in hash):
            final.append(find_closest_match(x, hash[x[0]]))
        else:
            continue
    return final
