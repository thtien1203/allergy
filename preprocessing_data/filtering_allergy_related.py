import pandas as pd
import re

df = pd.read_csv("./data/ChatDoctor_HealthCareMagic_100k.csv")

# remove instruction column
df = df.drop("instruction", axis=1)
df.to_csv("./data/filtering_allergy_related.csv", index=True)

# define keywords categories
core_allergy_keywords = [
    'allergy', 'allergic', 'allergies', 
    'anaphylaxis', 'anaphylactic'
]

# symptom_keywords = [
#     'hives', 'rash', 'itching', 'itchy', 'red', 'redness',
#     'swelling', 'swell', 'swollen',
#     'sneezing', 'sneeze', 'cough', 'congest',
#     'runny nose', 'stuffy nose', 'sore throat',
#     'watery eyes', 'itchy eyes', 'tear',
#     'wheezing', 'wheeze'
# ]

def is_allergy(text):
    # check if the text contains core allergy keywords
    if pd.isna(text):   
        return False    # no text to search
    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in core_allergy_keywords)

def has_symptoms(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    # return any(symptom in text_lower for symptom in symptom_keywords)

    # use regrex to match the complete word only
    symptom_patterns = [
    r'\bhives?\b', r'\brash(es)?\b', r'\bitch(ing|y|es)?\b', r'\bred\b',
    r'\bredness\b', r'\bbump(s)?\b', r'\bswell(ing|ed|s)?\b', r'\bswollen\b',
    r'\bsneez(e|ing|ed|es)?\b', r'\bwheez(e|ing|ed|es)?\b', r'\bcough(ing|ed|s)?\b',
    r'\bthroat\b', r'runny\s+nose', r'stuffy\s+nose', r'sore\s+throat',
    r'watery\s+eyes?', r'itchy\s+eyes?', r'\btear(s|y|ing)\b',
]
    return any(re.search(pattern, text_lower) for pattern in symptom_patterns)

# filter for allergy-related texts
df['is_allergy'] = df['input'].apply(is_allergy).astype(int)
allergy_df = df[df['is_allergy'] == 1].copy()

allergy_df['has_symptoms'] = allergy_df['input'].apply(has_symptoms).astype(int)

with_symptoms = allergy_df[allergy_df['has_symptoms'] == 1]
without_symptoms = allergy_df[allergy_df['has_symptoms'] == 0]
allergy_df = allergy_df.reset_index(drop=True)
print(f"\nOriginal dataset: {len(df)}")
print(f"Allergy-related: {len(allergy_df)}({len(allergy_df)/len(df)*100:.1f}%)")
print(f"\nOf the {len(allergy_df)} allergy-related texts:")
print(f"With symptom keywords: {len(with_symptoms)} ({len(with_symptoms)/len(allergy_df)*100:.1f}%)")
print(f"Without symptom keywords: {len(without_symptoms)} ({len(without_symptoms)/len(allergy_df)*100:.1f}%)")

# save filtered data
allergy_df.to_csv("./data/filtering_allergy_related.csv", index=True)

# keep only texts with symptoms
df_filtered = allergy_df[allergy_df['has_symptoms'] == 1].copy()
df_filtered = df_filtered.reset_index(drop=True)
df_filtered = df_filtered.drop(columns=['is_allergy'])
df_filtered.to_csv("./data/allergy_related_w_symptoms.csv", index=True)