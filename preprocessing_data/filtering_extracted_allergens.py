import pandas as pd
import re

df = pd.read_csv("./data/extracted_allergens.csv")

allergen = {
    # food allergens
    'peanut', 'peanuts', 'nut', 'cheese', 'dairy', 'eggs',
    'wheat', 'gluten', 'fish', 'shellfish', 'shrimp', 'crab',
    'sesame', 'cashew','chocolate', 'wine', 'creamed corn',
    'tomatoes', 'mushrooms', 'scallops', 'oysters', 'soy milk', 'soy beans',
    
    # drug allergens
    'penicillin', 'codeine', 'motrin', 'amoxicillin', 'azithromycin',
    
    # environmental allergens
    'pollen', 'pollens', 'dust', 'mold', 'mould', 
    'pet', 'dust mite', 'sulfas', 'climate',
    'latex', 'sulfur', 'pollution', 'evironmental', 
     
    # insect allergens
    'insects', 'bee', 'wasp', 'bite', 'insect bite', 'dogs', 'cats', 'mosquito', 
    # nuts, almonds, 'milk', egg, soy, 'ragweed',
    # allergy conditions
    # 'food', 'seasonal', 'contact', 'cat', 'bronchitis', 'nasal',  'ibuprofen','drug', 'grass', 'tree', 
    #  'iodine',
    # 'nickel','rhinitis', 'dermatitis', 'sinusitis', 'bronchitis',  'rhinosinusitis' 
    # 'azithromycin', 'walnut', 'animal', 'antibiotic', 'antibiotics', 'tree nut',  'chicken', 'alcohol', 'metal', 'amoxicillin', 
    # common extracted allergen-related terms (condition)
    # walnut,'sting', 'dander','fur','asthma', 'chronic', 'rhinitis', 'dermatitis', 'sinusitis', 'bronchitis', 'seafood', 
}
normalization_map = {
    'eggs': 'egg',
    'nuts': 'nut',
    'peanuts': 'peanut',
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
df['patient'] = df['patient_extracted_word'].apply(is_valid_allergen)
df['doctor'] = df['doctor_extracted_word'].apply(is_valid_allergen)

# remove rows where both are invalid or empty
df_filtered = df[df['patient'] | df['doctor']].copy()
df_filtered['patient_cleaned'] = df_filtered['patient_extracted_word'].apply(clean_allergen)
df_filtered['doctor_cleaned'] = df_filtered['doctor_extracted_word'].apply(clean_allergen)
# df['patient_cleaned'] = df['patient_extracted_word'].apply(clean_allergen)
# df['doctor_cleaned'] = df['doctor_extracted_word'].apply(clean_allergen)

results = []
for idx, row in df_filtered.iterrows():
    patient = row['patient_cleaned']
    doctor = row['doctor_cleaned']
    # take from patient if doctor is empty
    if pd.notna(patient) and pd.isna(doctor):
        results.append({'allergens': patient})
    # take from doctor if patient is empty
    elif pd.isna(patient) and pd.notna(doctor):
        results.append({'allergens': doctor})
    # if both have extracted words 
    elif pd.notna(patient) and pd.notna(doctor):
        # take patient
        results.append({'allergens': patient})
 
df_results = pd.DataFrame(results)
df_filtered= pd.concat([df_filtered.reset_index(drop=True), df_results], axis=1)
df_final = df_filtered[['Unnamed: 0', 'input', 'output', 'has_symptoms', 'allergens']].copy()
df_final['Unnamed: 0'] = range(len(df_final))
# df_final["allergens"] = df_final["allergens"].fillna("None")
df_final.to_csv("./data/filtering_extracted_allergens.csv", index=False)
