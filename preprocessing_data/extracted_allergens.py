import pandas as pd
import re

df = pd.read_csv("./data/allergy_related_w_symptoms.csv")

allergen = {
    # food allergens
    'peanut', 'peanuts', 'nut', 'nuts', 'milk', 'dairy', 'egg', 'eggs',
    'wheat', 'gluten', 'soy', 'fish', 'shellfish', 'shrimp', 'crab',
    'sesame', 'tree nut', 'almond', 'cashew', 'walnut', 'chocolate', 'chicken', 'alcohol', 'wine',
    'tomatoes', 'mushrooms', 'scallops', 'oysters',
    
    # drug allergens
    'penicillin', 'ibuprofen', 'antibiotic', 'antibiotics',
    'drug', 'amoxicillin', 'codeine', 'motrin', 
    
    # environmental allergens
    'pollen', 'pollens', 'dust', 'mold', 'mould', 'grass', 'tree', 'ragweed',
    'dog', 'pet', 'animal', 'mite', 'sulfas',
    'latex', 'sulfur', 'pollution', 'evironmental',
    
    # insect allergens
    'bee', 'wasp', 'insect bite', 'dogs', 'cats', 'mosquito',
    
    # allergy conditions
    # 'food', 'seasonal', 'contact', 'cat', 'bronchitis', 'nasal',
    'nickel', 'iodine', 'metal'
    
    # common extracted allergen-related terms (condition)
    # 'seafood','sting', 'dander','fur','asthma', 'chronic', 'rhinitis', 'dermatitis', 'sinusitis', 'bronchitis'
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
