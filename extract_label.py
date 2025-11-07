"""
Preliminary Data Analysis - FOR PROPOSAL
=========================================
Complete statistical analysis of cleaned allergen dataset
for project proposal requirements.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

print("="*80)
print("PRELIMINARY DATA ANALYSIS - CLEANED DATASET")
print("="*80)

# Load cleaned data
df = pd.read_csv("./data/filtering_extracted_allergens.csv")

# ============================================================================
# 1. ESSENTIAL DATA STATISTICS
# ============================================================================
print("\n1. ESSENTIAL DATA STATISTICS")
print("-" * 80)
print(f"Total samples: {len(df)}")
print(f"Samples with allergens: {df['allergen'].notna().sum()}")
print(f"Samples with symptoms: {df['has_symptoms'].sum()}")

# ============================================================================
# 2. ALLERGEN TYPE MAPPING (Cleaned - No conditions!)
# ============================================================================
allergen_type_map = {
    # Food allergens
    'peanut': 'food', 'nut': 'food', 'milk': 'food', 'dairy': 'food', 
    'egg': 'food', 'wheat': 'food', 'gluten': 'food', 'soy': 'food',
    'fish': 'food', 'shellfish': 'food', 'shrimp': 'food', 'crab': 'food',
    'sesame': 'food', 'almond': 'food', 'cashew': 'food', 'walnut': 'food',
    'chocolate': 'food', 'chicken': 'food', 'alcohol': 'food', 'wine': 'food',
    'seafood': 'food', 'tomatoes': 'food', 'mushrooms': 'food',
    
    # Drug allergens
    'penicillin': 'drug', 'ibuprofen': 'drug', 'antibiotic': 'drug',
    'amoxicillin': 'drug', 'codeine': 'drug', 'motrin': 'drug',
    'aspirin': 'drug', 'sulfa': 'drug',
    
    # Environmental allergens
    'pollen': 'environmental', 'dust': 'environmental', 'mold': 'environmental',
    'grass': 'environmental', 'tree': 'environmental', 'ragweed': 'environmental',
    'dander': 'environmental', 'fur': 'environmental', 'mite': 'environmental',
    'pollution': 'environmental', 'sulfur': 'environmental',
    
    # Pet allergens (subcategory of environmental)
    'cat': 'pet', 'dog': 'pet', 'animal': 'pet', 'pet': 'pet',
    
    # Insect allergens
    'bee': 'insect', 'wasp': 'insect', 'sting': 'insect', 'mosquito': 'insect',
    
    # Contact allergens
    'nickel': 'contact', 'latex': 'contact', 'metal': 'contact', 'iodine': 'contact',
}

def get_allergy_types(allergen_str):
    """Get all allergy types from comma-separated allergens"""
    if pd.isna(allergen_str):
        return []
    allergens = [a.strip().lower() for a in str(allergen_str).split(',')]
    types = []
    for allergen in allergens:
        if allergen in allergen_type_map:
            allergy_type = allergen_type_map[allergen]
            if allergy_type not in types:
                types.append(allergy_type)
    return types

# Get allergy types for each row
df['allergy_types'] = df['allergen'].apply(get_allergy_types)
df['num_allergy_types'] = df['allergy_types'].apply(len)

# Count samples per type
all_types = []
for types_list in df['allergy_types']:
    all_types.extend(types_list)

type_counts = Counter(all_types)

# ============================================================================
# 2. CLASS DISTRIBUTION (Allergy Types)
# ============================================================================
print("\n2. CLASS DISTRIBUTION (Allergy Types)")
print("-" * 80)
total_labeled = sum(type_counts.values())

print("Distribution of allergy categories:")
for allergy_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_labeled) * 100
    print(f"  {allergy_type:20s}: {count:4d} samples ({percentage:5.1f}%)")

print(f"\nTotal type labels: {total_labeled}")
print(f"Unique samples: {len(df)}")
print(f"Samples with multiple allergy types: {(df['num_allergy_types'] > 1).sum()} ({(df['num_allergy_types'] > 1).sum()/len(df)*100:.1f}%)")
print(f"Samples with single allergy type: {(df['num_allergy_types'] == 1).sum()} ({(df['num_allergy_types'] == 1).sum()/len(df)*100:.1f}%)")

# Calculate imbalance ratio
if len(type_counts) > 0:
    most_common = max(type_counts.values())
    least_common = min(type_counts.values())
    imbalance_ratio = most_common / least_common
    print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"  Most common: {max(type_counts, key=type_counts.get)} ({most_common} samples)")
    print(f"  Least common: {min(type_counts, key=type_counts.get)} ({least_common} samples)")

# ============================================================================
# 3. VOCABULARY AND TEXT COMPLEXITY
# ============================================================================
print("\n3. VOCABULARY AND TEXT COMPLEXITY")
print("-" * 80)

# Calculate statistics for input text
df['input_word_count'] = df['input'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
df['input_char_count'] = df['input'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
df['input_sentence_count'] = df['input'].apply(lambda x: max(1, len(re.split(r'[.!?]+', str(x))) - 1) if pd.notna(x) else 1)
df['avg_words_per_sentence'] = df['input_word_count'] / df['input_sentence_count']

print(f"Query length statistics:")
print(f"  Average: {df['input_word_count'].mean():.1f} words")
print(f"  Median: {df['input_word_count'].median():.1f} words")
print(f"  Min: {df['input_word_count'].min()} words")
print(f"  Max: {df['input_word_count'].max()} words")
print(f"  Std dev: {df['input_word_count'].std():.1f} words")

print(f"\nText complexity:")
print(f"  Average characters per query: {df['input_char_count'].mean():.1f}")
print(f"  Average sentences per query: {df['input_sentence_count'].mean():.1f}")
print(f"  Average words per sentence: {df['avg_words_per_sentence'].mean():.1f}")

# Vocabulary size
all_words = []
for text in df['input'].dropna():
    words = str(text).lower().split()
    all_words.extend(words)

vocab_size = len(set(all_words))
total_words = len(all_words)
print(f"\nVocabulary statistics:")
print(f"  Total vocabulary size: {vocab_size:,} unique tokens")
print(f"  Total word count: {total_words:,} words")
print(f"  Average vocabulary per query: {total_words/len(df):.1f} words")

# ============================================================================
# 4. LINGUISTIC FEATURES - Symptom Keywords
# ============================================================================
print("\n4. LINGUISTIC FEATURES - Most Common Symptom Keywords")
print("-" * 80)

symptom_keywords = [
    'itching', 'itch', 'rash', 'hives', 'swelling', 'swell', 'sneezing', 'sneeze',
    'runny', 'nose', 'watery', 'eyes', 'cough', 'throat', 'breathing', 'wheeze',
    'red', 'bump', 'reaction', 'pain', 'burning', 'irritation', 'congestion'
]

symptom_counts = Counter()
for text in df['input'].dropna():
    text_lower = str(text).lower()
    for symptom in symptom_keywords:
        if symptom in text_lower:
            symptom_counts[symptom] += 1

print("Top 15 most mentioned symptoms:")
for symptom, count in symptom_counts.most_common(15):
    percentage = (count / len(df)) * 100
    print(f"  {symptom:15s}: {count:4d} occurrences ({percentage:4.1f}% of queries)")

# ============================================================================
# 5. ALLERGEN EXTRACTION STATISTICS
# ============================================================================
print("\n5. ALLERGEN EXTRACTION STATISTICS")
print("-" * 80)

# Count individual allergens
all_allergens = []
for allergen_str in df['allergen'].dropna():
    allergens = [a.strip().lower() for a in str(allergen_str).split(',')]
    all_allergens.extend(allergens)

allergen_counts = Counter(all_allergens)
print(f"Total allergen mentions: {len(all_allergens)}")
print(f"Unique allergens extracted: {len(allergen_counts)}")
print(f"Average allergens per sample: {len(all_allergens)/len(df):.2f}")

print("\nTop 20 most common allergens:")
for allergen, count in allergen_counts.most_common(20):
    print(f"  {allergen:20s}: {count:4d} mentions")

# Allergens per query distribution
df['num_allergens'] = df['allergen'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
print(f"\nAllergens per query statistics:")
print(f"  Average: {df['num_allergens'].mean():.2f}")
print(f"  Median: {df['num_allergens'].median():.1f}")
print(f"  Max: {df['num_allergens'].max()}")
print(f"  Single allergen: {(df['num_allergens'] == 1).sum()} samples ({(df['num_allergens'] == 1).sum()/len(df)*100:.1f}%)")
print(f"  Multiple allergens: {(df['num_allergens'] > 1).sum()} samples ({(df['num_allergens'] > 1).sum()/len(df)*100:.1f}%)")

# ============================================================================
# 6. ADDITIONAL DATASET CHARACTERISTICS
# ============================================================================
print("\n6. ADDITIONAL DATASET CHARACTERISTICS")
print("-" * 80)

# Query length distribution
short = (df['input_word_count'] <= 50).sum()
medium = ((df['input_word_count'] > 50) & (df['input_word_count'] <= 150)).sum()
long = (df['input_word_count'] > 150).sum()

print(f"Query length categories:")
print(f"  Short (≤50 words): {short} ({short/len(df)*100:.1f}%)")
print(f"  Medium (51-150 words): {medium} ({medium/len(df)*100:.1f}%)")
print(f"  Long (>150 words): {long} ({long/len(df)*100:.1f}%)")

# Allergen diversity per category
print(f"\nAllergen diversity by category:")
for allergy_type in sorted(type_counts.keys()):
    allergens_in_type = [a for a in all_allergens if allergen_type_map.get(a) == allergy_type]
    unique_in_type = len(set(allergens_in_type))
    print(f"  {allergy_type:20s}: {unique_in_type} unique allergens")

# ============================================================================
# 7. DATA QUALITY METRICS
# ============================================================================
print("\n7. DATA QUALITY METRICS")
print("-" * 80)

print(f"Completeness:")
print(f"  Samples with allergens: {df['allergen'].notna().sum()} ({df['allergen'].notna().sum()/len(df)*100:.1f}%)")
print(f"  Samples with symptoms: {df['has_symptoms'].sum()} ({df['has_symptoms'].sum()/len(df)*100:.1f}%)")
print(f"  Complete samples (allergen + symptoms): {(df['allergen'].notna() & df['has_symptoms']).sum()} ({(df['allergen'].notna() & df['has_symptoms']).sum()/len(df)*100:.1f}%)")

# ============================================================================
# SUMMARY FOR PROPOSAL
# ============================================================================
print("\n" + "="*80)
print("SUMMARY FOR PROPOSAL")
print("="*80)

print(f"""
Dataset Overview:
- Total samples: {len(df)}
- Unique allergens: {len(allergen_counts)}
- Allergy categories: {len(type_counts)}

Text Complexity:
- Average query length: {df['input_word_count'].mean():.1f} words
- Average sentences per query: {df['input_sentence_count'].mean():.1f}
- Average words per sentence: {df['avg_words_per_sentence'].mean():.1f}
- Vocabulary size: {vocab_size:,} unique tokens

Class Distribution:
- Most common: {max(type_counts, key=type_counts.get)} ({max(type_counts.values())} samples, {max(type_counts.values())/total_labeled*100:.1f}%)
- Least common: {min(type_counts, key=type_counts.get)} ({min(type_counts.values())} samples, {min(type_counts.values())/total_labeled*100:.1f}%)
- Imbalance ratio: {imbalance_ratio:.1f}:1
- Multi-label samples: {(df['num_allergy_types'] > 1).sum()} ({(df['num_allergy_types'] > 1).sum()/len(df)*100:.1f}%)

Top Symptoms: {', '.join([s for s, _ in symptom_counts.most_common(5)])}
Top Allergens: {', '.join([a for a, _ in allergen_counts.most_common(5)])}
""")

print("="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)

# Save summary to file
with open("./data/preliminary_analysis_summary.txt", "w") as f:
    f.write("PRELIMINARY DATA ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Unique allergens: {len(allergen_counts)}\n")
    f.write(f"Allergy categories: {len(type_counts)}\n")
    f.write(f"Average query length: {df['input_word_count'].mean():.1f} words\n")
    f.write(f"Vocabulary size: {vocab_size:,} tokens\n")
    f.write(f"Class imbalance: {imbalance_ratio:.1f}:1\n")
    f.write(f"Multi-label samples: {(df['num_allergy_types'] > 1).sum()}\n")

print("\n✓ Saved summary: preliminary_analysis_summary.txt")