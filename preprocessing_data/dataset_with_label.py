import pandas as pd
from collections import Counter

# load the allergens with labels
df_labels = pd.read_csv("./data/word_embeddings_labeled.csv")
df_labels["allergens"] = df_labels["allergens"].str.lower().str.strip()

# load the original extracted allergens
df_filtered = pd.read_csv("./data/filtering_extracted_allergens.csv", keep_default_na=False)
map_labels = dict(zip(df_labels["allergens"], df_labels["labels"]))

# count allergen frequencies in dataset
all_allergens = []
for allergens_str in df_filtered['allergens'].dropna():
    allergens = [a.strip().lower() for a in str(allergens_str).split(',')]
    all_allergens.extend(allergens)
allergen_counts = Counter(all_allergens)
# calculate total frequency for each group
category_total = Counter()
for allergen, count in allergen_counts.items():
    category = map_labels.get(allergen, 'Unknown')
    category_total[category] += count
print("\nCategory frequencies (from data):")
for category, count in category_total.most_common():
    print(f"{category}: {count}")

# create priority based on frequency
sorted_categories = [cat for cat, _ in category_total.most_common()]
priority_order = {cat: idx + 1 for idx, cat in enumerate(sorted_categories)}
print("\nPriority order (based on data frequency):")
for cat, priority in sorted(priority_order.items(), key=lambda x: x[1]):
    print(f"{cat}: {priority} (count: {category_total.get(cat, 0)})")

# apply single label for each datapoint
def get_labels(cell):
    if pd.isna(cell):
        return 'None'
    allergens = [a.strip().lower() for a in str(cell).split(",")]
    categories = [map_labels.get(a, "Unknown") for a in allergens]
    category_counts = Counter(categories)   # count occurences
    max_count = max(category_counts.values())
    # get all categories with max count
    top_categories = [cat for cat, count in category_counts.items() if count == max_count]
    # if only one category has max count, return it
    if len(top_categories) == 1:
        return top_categories[0]
    # use priority based on overall frequency if needed (only work if 2 groups in each datapoint)
    top_categories_sorted = sorted(top_categories, key=lambda x: priority_order.get(x, 99))
    return top_categories_sorted[0]

df_filtered["labels"] = df_filtered["allergens"].apply(get_labels)
df_filtered.to_csv("./data/dataset_with_labeled.csv", index=False)

# save all labels into one file
# get unique labels from clustering file
unique_labels = sorted(df_labels["labels"].dropna().unique().tolist())
all_labels = sorted(set(unique_labels))
df_all_labels = pd.DataFrame({"labels": all_labels})
df_all_labels.to_csv("./data/labels.csv", index=False)
