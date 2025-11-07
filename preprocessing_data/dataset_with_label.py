import pandas as pd

# load the allergens with labels
df_labels = pd.read_csv("./data/word_embeddings_labeled.csv")
df_labels["allergens"] = df_labels["allergens"].str.lower().str.strip()

# load the original extracted allergens
df_filtered = pd.read_csv("./data/filtering_extracted_allergens.csv", keep_default_na=False)
map_labels = dict(zip(df_labels["allergens"], df_labels["labels"]))

# get label for each allergen cell
def get_labels(cell):
    if pd.isna(cell):
        return 'None'
    allergens = [a.strip().lower() for a in str(cell).split(",")]
    cats = [map_labels.get(a, "Unknown") for a in allergens]
    unique_cats = sorted(set(cats)) # remove duplicates
    return ", ".join(unique_cats)
df_filtered["labels"] = df_filtered["allergens"].apply(get_labels)
df_filtered.to_csv("./data/dataset_with_labeled.csv", index=False)

# save all labels into one file
extra_labels = ['chronic', 'rhinitis', 'dermatitis', 'sinusitis', 'bronchitis']
# get unique labels from clustering file
unique_labels = sorted(df_labels["labels"].dropna().unique().tolist())
# merge and remove duplicates
all_labels = sorted(set(unique_labels + extra_labels))
df_all_labels = pd.DataFrame({"labels": all_labels})
df_all_labels.to_csv("./data/labels.csv", index=False)
