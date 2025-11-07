import pandas as pd
import re

df = pd.read_csv("./data/extracted_allergens.csv")

allergen = {
    # food allergens
    'peanut', 'peanuts', 'nut', 'nuts', 'milk', 'dairy', 'egg', 'eggs',
    'wheat', 'gluten', 'soy', 'fish', 'shellfish', 'shrimp', 'crab',
    'sesame', 'tree nut', 'almond', 'cashew', 'walnut', 'chocolate', 'chicken', 'alcohol', 'wine',
    'tomatoes', 'mushrooms', 'oysters', 'scallops',
    
    # drug allergens
    'penicillin', 'ibuprofen', 'antibiotic', 'antibiotics',
    'drug', 'amoxicillin', 'codeine', 'motrin', 
    
    # environmental allergens
    'pollen', 'pollens', 'dust', 'mold', 'mould', 'grass', 'tree', 'ragweed',
    'dog', 'pet', 'animal', 'mite', 'sulfas',
    'latex', 'sulfur', 'pollution', 'evironmental',
    
    # insect allergens
    'bee', 'wasp', 'insect bite','dogs', 'cats', 'mosquito',
    
    # allergy conditions
    # 'food', 'seasonal', 'contact', 'cat', 'bronchitis', 'nasal',
    'nickel', 'iodine', 'metal'
    
    # common extracted allergen-related terms (condition)
    #  'seafood','sting', 'dander','fur', 'asthma', 'chronic', 'rhinitis', 'dermatitis', 'sinus', 'bronchitis'
}
normalization_map = {
    'eggs': 'egg',
    'nuts': 'nut',
    'peanuts': 'peanut',
    'antibiotics': 'antibiotic',
    'pollens': 'pollen',
    'dogs': 'dog',
    'mould': 'mold',
}
def normalize_word(word):
    word = word.strip().lower()
    return normalization_map.get(word, word)
def is_valid_allergen(word):
    if pd.isna(word) or word == ' ':
        return False
    word_clean = str(word).lower().strip()
    # return word_clean in allergen
    if ',' in word_clean:
        words = [w.strip() for w in word_clean.split(',')]
        return any(w in allergen for w in words)
    else:
        return word_clean in allergen

def clean_allergen(word):
    if pd.isna(word) or word == ' ':
        return None
    word_clean = str(word).lower().strip()
    if ',' in word_clean:
        words = [w.strip() for w in word_clean.split(',')]
        valid_words = [w for w in words if w in allergen]
        if valid_words:
            return ', '.join(valid_words)
        else:
            return None
    else:
        if word_clean in allergen:
            return word_clean
        else:
            return None
# check validity of extractions
# df['patient'] = df['patient_extracted_word'].apply(is_valid_allergen)
# df['doctor'] = df['doctor_extracted_word'].apply(is_valid_allergen)

# remove rows where both are invalid or empty
# df_filtered = df[df['patient'] | df['doctor']].copy()
# df_filtered['patient_cleaned'] = df_filtered['patient_extracted_word'].apply(clean_allergen)
# df_filtered['doctor_cleaned'] = df_filtered['doctor_extracted_word'].apply(clean_allergen)
df['patient_cleaned'] = df['patient_extracted_word'].apply(clean_allergen)
df['doctor_cleaned'] = df['doctor_extracted_word'].apply(clean_allergen)

results = []
for idx, row in df.iterrows():
    patient = row['patient_cleaned']
    doctor = row['doctor_cleaned']
    # if both invalid or empty, keep sample but set allergen to None
    if pd.isna(patient) and pd.isna(doctor):
        results.append({'allergens': "None"})
    # only one has extracted word 
    elif pd.notna(patient) and pd.isna(doctor):
        # only patient has valid allergen
        results.append({'allergens': patient})
    elif pd.isna(patient) and pd.notna(doctor):
        # only doctor has valid allergen
        results.append({'allergens': doctor})
    # if both have extracted words 
    elif pd.notna(patient) and pd.notna(doctor):
        # if both have the same allergen type, keep one
        # combine both and remove duplicates
        patient_words = [normalize_word(w) for w in patient.split(',')]
        doctor_words = [normalize_word(w)  for w in doctor.split(',')]
        combined = []
        for word in patient_words + doctor_words:
            if word not in combined:
                combined.append(word)
            
        results.append({'allergens': ', '.join(combined)})
    # else:
    #     results.append({'allergens': None})
df_results = pd.DataFrame(results)
df_final = pd.concat([df.reset_index(drop=True), df_results], axis=1)
df_final = df_final[['Unnamed: 0', 'input', 'output', 'has_symptoms', 'allergens']].copy()
df_final['Unnamed: 0'] = range(len(df_final))
df_final["allergens"] = df_final["allergens"].fillna("None")
df_final.to_csv("./data/filtering_extracted_allergens.csv", index=False)

# from collections import Counter
# all_allergens = []
# for allergen_str in df_final['allergens'].dropna():
#     allergens = [a.strip().lower() for a in str(allergen_str).split(',')]
#     all_allergens.extend(allergens)

# allergen_counts = Counter(all_allergens)
# print(f"\nSummary:")
# print(f"Total samples: {len(df_final)}")
# print(f"Unique allergens: {len(allergen_counts)}")
# print(f"Total allergen mentions: {sum(allergen_counts.values())}")
