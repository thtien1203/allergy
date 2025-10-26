import pandas as pd

df = pd.read_csv("./data/ChatDoctor_HealthCareMagic_100k.csv")

# remove instruction column
df = df.drop("instruction", axis=1)
df.to_csv("./data/preprocessed_data.csv", index=True)

# define keywords categories
core_allergy_keywords = [
    'allergy', 'allergic', 'allergies', 
    'anaphylaxis', 'anaphylactic'
]

symptom_keywords = [
    'hives', 'rash', 'itching', 'itchy',
    'swelling', 'swell', 'swollen',
    'sneezing', 'sneeze',
    'runny nose', 'stuffy nose', 'sore throat',
    'watery eyes', 'itchy eyes',
    'wheezing', 'wheeze'
]

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
    return any(symptom in text_lower for symptom in symptom_keywords)

# filter for allergy-related texts
df['is_allergy'] = df['input'].apply(is_allergy).astype(int)
allergy_df = df[df['is_allergy'] == 1].copy()

allergy_df['has_symptoms'] = allergy_df['input'].apply(has_symptoms).astype(int)

with_symptoms = allergy_df[allergy_df['has_symptoms'] == 1]
without_symptoms = allergy_df[allergy_df['has_symptoms'] == 0]
allergy_df = allergy_df.reset_index(drop=True)
print(f"\nOriginal dataset: {len(df)} rows")
print(f"Allergy-related: {len(allergy_df)} rows ({len(allergy_df)/len(df)*100:.1f}%)")
print(f"\nOf the {len(allergy_df)} allergy-related texts:")
print(f"With symptom keywords: {len(with_symptoms)} ({len(with_symptoms)/len(allergy_df)*100:.1f}%)")
print(f"Without symptom keywords: {len(without_symptoms)} ({len(without_symptoms)/len(allergy_df)*100:.1f}%)")

# save filtered data
allergy_df.to_csv("./data/preprocessed_data.csv", index=True)