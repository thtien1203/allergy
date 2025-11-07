import pandas as pd
import numpy as np
import spacy
import json
from collections import Counter
import gensim.downloader as api

df = pd.read_csv("./data/filtering_extracted_allergens.csv")
# get unique allergens
all_allergens = []
for allergens_str in df['allergens'].dropna():
    allergens = [a.strip().lower() for a in str(allergens_str).split(',')]
    all_allergens.extend(allergens)

allergen_counts = Counter(all_allergens)
unique_allergens = list(allergen_counts.keys())
print(f"Found {len(unique_allergens)} unique allergens")

# word embeddings
nlp = spacy.load("en_core_web_lg")
embeddings = []
valid_allergens = []
for allergens in unique_allergens:
    doc = nlp(allergens)
    if doc.has_vector:
        embeddings.append(doc.vector)
        valid_allergens.append(allergens)
embeddings = np.array(embeddings)
# save as vector
vectors_as_text = [json.dumps(vec.tolist()) for vec in embeddings]

# create dataframe
emb_df = pd.DataFrame({
    "allergens": valid_allergens,
    "word_embeddings": vectors_as_text
})
emb_df.to_csv("./data/word_embeddings.csv", index=False)

