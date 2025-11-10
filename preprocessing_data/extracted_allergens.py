import pandas as pd
import re

df = pd.read_csv("./data/allergy_related_w_symptoms.csv")

allergen = {
    # food allergens
    'peanut', 'peanuts', 'nut', 'cheese', 'dairy', 'eggs',
    'wheat', 'gluten', 'fish', 'shellfish', 'shrimp', 'crab',
    'sesame', 'cashew','chocolate', 'wine', 
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
    'iodine',
    #  'nickel','rhinitis', 'dermatitis', 'sinusitis', 'bronchitis',  'rhinosinusitis' 
    # 'azithromycin', 'walnut', 'animal', 'antibiotic', 'antibiotics', 'tree nut',  'chicken', 'alcohol', 'metal', 'amoxicillin', 
    # common extracted allergen-related terms (condition)
    # walnut,'sting', 'dander','fur','asthma', 'chronic', 'rhinitis', 'dermatitis', 'sinusitis', 'bronchitis', 'seafood', 
}
def extract_allergens_with_regex(text):
    """
    Search text for any allergen words using regex
    Returns: list of found allergens (or None if none found)
    """
    if pd.isna(text):
        return None
    
    text_lower = str(text).lower()
    
    # Create regex pattern from allergen list
    # Sort by length (longest first) to match "tree nut" before "nut"
    allergen_sorted = sorted(allergen, key=len, reverse=True)
    
    # Escape special regex characters and join with |
    pattern = r'\b(' + '|'.join(re.escape(a) for a in allergen_sorted) + r')\b'
    
    # Find all matches
    matches = re.findall(pattern, text_lower)
    
    if matches:
        # Remove duplicates but keep order
        unique_matches = []
        for match in matches:
            if match not in unique_matches:
                unique_matches.append(match)
        return ', '.join(unique_matches)
    else:
        return None
    
df['patient_extracted_word'] = df['input'].apply(extract_allergens_with_regex)
df['doctor_extracted_word'] = df['output'].apply(extract_allergens_with_regex)
df.to_csv("./data/extracted_allergens.csv", index=False)
